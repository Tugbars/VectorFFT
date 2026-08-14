(* schedule.ml — Starve–Retire (SR) list scheduling: the production
 *               instruction scheduler, with a Goodman-Hsu pressure
 *               switch and port-class balancing.
 *
 * === WHY THE GENERATOR SCHEDULES AT ALL ===
 *   A codelet is one straight-line block with hundreds of live SIMD
 *   values competing for 16 YMM / 32 ZMM registers. For such blocks the
 *   EMISSION ORDER largely determines peak register pressure, and peak
 *   pressure determines stack spills — the dominant cost the compiler
 *   cannot undo well on its own (measured: gcc re-schedules toward
 *   MORE pressure under stress; see the regalloc.ml history). So the
 *   generator computes a good order itself. gcc mostly preserves it,
 *   and the opt-in fence machinery in regalloc.ml can pin it exactly.
 *
 * === 1. THE ALGORITHM: STARVE–RETIRE ===
 *   Ready-list scheduling over the hash-consed DAG. Ablation (doc 69)
 *   shows the algorithm IS two rules about value lifetimes, not the
 *   classical priority keys:
 *
 *   - STARVE — starvation-gated load admission (late birth): a load
 *     fires ONLY when no arithmetic is ready, and loads fire strictly
 *     in source (tag) order. Eager load fronts inflate the live set
 *     for no ILP gain; starvation gating delays each value's birth to
 *     the last useful moment, and source order keeps the access
 *     stream sequential for the hardware prefetcher. This rule is
 *     worth ~8x on model Belady traffic: with it, even tag-only
 *     ordering scores 43; without it, every priority-key combination
 *     collapses to 230-272.
 *
 *   - RETIRE — eager sink retirement (early death): a ready sink (a
 *     node nothing else consumes — it only feeds an output store)
 *     preempts all other ranking. Firing it the moment its inputs
 *     exist kills those inputs on the spot. Second load-bearing rule:
 *     traffic 43 -> 28 in the same ablation. Concretely, DIF prime
 *     codelets' post-multiply cmul nodes ARE the outputs; under pure
 *     critical-path ranking they fire last and pin their inputs live
 *     across the whole body — with RETIRE, R=17 t1_dif drops from 176
 *     to ~115 vmovapd, matching the hand-written codelet. DIT is
 *     naturally unaffected (its cmuls are sources, not sinks).
 *
 *   Conjecture (doc 69): SR is the scheduling-side dual of Belady MIN
 *   eviction on butterfly DAGs — late birth + early death minimizes
 *   the same traffic objective the optimal evictor minimizes.
 *
 * === 2. TIEBREAK KEYS (near-neutral refinements) ===
 *   Among ready arithmetic, ties resolve by cp_dist DESC, su_num ASC,
 *   tag ASC. Ablated at +-2 traffic units with radix-dependent sign —
 *   kept for determinism and latency shaping, but they are NOT where
 *   the schedule quality comes from:
 *
 *   - cp_dist (compute_cp_dist): critical-path distance to a sink in
 *     CYCLES from the Uarch latency table — backward DP over the DAG
 *     in reverse-topological order (tag-descending, since hash-cons
 *     tags are assigned bottom-up).
 *   - su_num (compute_su_number): the Sethi-Ullman register-need
 *     label, generalized k-ary for Cmul / Fma atoms (sort child
 *     labels descending, label = max over i of child_i + i). Exact on
 *     trees; over-counts shared DAG subtrees — the conservative bias.
 *   - tag ASC last makes every schedule deterministic and
 *     regeneration-stable.
 *
 *   Function and flag names (su_schedule, --su) predate the SR naming
 *   and are retained.
 *
 * === 3. THE GOODMAN-HSU PRESSURE SWITCH (~gh:true) ===
 *   Latency priority and pressure priority want different schedules;
 *   Goodman-Hsu's answer is to switch keys based on the live count:
 *   - LATENCY MODE while live <= threshold: the base priority above.
 *   - PRESSURE MODE while live > threshold: pick the candidate with
 *     the lowest delta = births - kills, i.e. the op that frees the
 *     most registers relative to what it allocates.
 *   The threshold comes from the Uarch profile (pressure_threshold:
 *   24 on AVX-512, 12 on AVX2, slack reserved for cmul/FMA scratch);
 *   VFFT_GH_THRESHOLD overrides it for experiments. The recipe layer
 *   auto-enables gh on AVX2 at R >= 32, where it measures 4-8% over
 *   the base recipe and is a no-op on AVX-512.
 *
 * === 4. PORT-CLASS BALANCING (tiebreaker) ===
 *   On Ice Lake server, Mul / FMA / Cmul dispatch only to ports 0-1
 *   while Add / Sub / Neg can also use port 5; a mul-heavy stretch
 *   starves P5 while P0/P1 back up. gcc-11's post-RA scheduler earns
 *   ~2.5-3.4% (llvm-mca, R=32..256 c2c) by re-interleaving. When
 *   cp_dist is tied within a small window, we prefer the op whose
 *   port class is behind — recovering that interleave at the source
 *   without disturbing the critical path.
 *
 * === 5. SUBSET SCHEDULING (the spill recipe's entry point) ===
 *   su_schedule_subset schedules a pre-selected subset (PASS 1 or
 *   PASS 2 of a blocked codelet) while treating out-of-subset
 *   predecessors — original inputs, reloaded spill values, hoisted
 *   constants — as already available. cp_dist is still computed over
 *   the full reachable graph so depth information survives the cut.
 *   This is the granularity the cluster-spill recipe emits at, and
 *   the baseline bb.ml's branch-and-bound refiner starts from.
 *
 * === 6. SCHEDULE-SEARCH HOOKS ===
 *   VFFT_SCHED_DUMP writes the chosen order plus the DAG edges;
 *   VFFT_SCHED_ORDER replays an explicit tag order. Together they
 *   form the interface the external schedule annealer drives to
 *   search order-space beyond the heuristic (see new-docs 66 and 68).
 *   Both default off; output is byte-identical without them.
 * ------------------------------------------------------------------
 * MODULE CARD (schedule.ml — grep "MODULE CARD" for the full set)
 * ROLE: The production scheduler: SU + GH list scheduling and its
 * cp-dist / su-number analyses, port-balance tiebreak, search hooks.
 * PIPELINE: Algsimp DAG -> a schedule -> Regalloc / Emit renderers
 * PUBLIC SURFACE (measured): emit_c: su_schedule,
 * su_schedule_subset; bb: su_schedule_subset, compute_cp_dist;
 * codelet_oop: su_schedule_subset.
 * DEPS: Algsimp (open — the heaviest client of the IR), Expr, Uarch.
 * ENV: VFFT_GH_THRESHOLD, VFFT_SCHED_ORDER, VFFT_SCHED_DUMP.
 * DOCS: new-docs 66 (annealer architecture), 68 (bicriteria
 * formulation), 69 (SU certification), 22 (Bb comparison).
 * ------------------------------------------------------------------
 *)

(* M1: `open Algsimp` removed — only Ir's names were used here; the
   dependency on the pass layer was an artifact of the include chain. *)
open Ir (* M1: names formerly re-exported through the chain *)

(* ═══════════════════════════════════════════════════════════════
 *  SCHEDULE WISDOM PLUMBING (order-injection provenance + staleness)
 * ═══════════════════════════════════════════════════════════════
 *
 * The env-gated dump/inject knobs (VFFT_SCHED_DUMP / VFFT_SCHED_ORDER)
 * are the interface the offline schedule search drives. Wisdom-grade
 * use adds three requirements the raw knobs lack:
 *
 *   1. RESOLUTION. emit_c resolves the per-codelet order source into
 *      `order_source` before scheduling: explicit VFFT_SCHED_ORDER
 *      wins; else VFFT_SCHED_WISDOM/<codelet symbol> (the symbol
 *      encodes R, family, direction, ISA — the wisdom key). When the
 *      ref is unset the injectors fall back to the env var directly,
 *      so standalone drivers keep their existing behavior.
 *
 *   2. STALENESS. An order file may carry a "#dagsig <md5>" first
 *      line — the digest of the canonical DAG the order was searched
 *      against. On mismatch the injector REFUSES the file, logs the
 *      event, and emits su's order. subset_key (Hashtbl.hash) only
 *      NAMES files; the dagsig is what turns a stale match from
 *      silent into detected-and-reported. Files without a dagsig
 *      (legacy search output) still inject, logged as unverified.
 *
 *   3. PROVENANCE. Every accept/refuse is recorded in
 *      `injection_log`; emit_c splices the log into the generated
 *      file, so an annealed codelet is distinguishable from stock
 *      (previously an injected codelet stamped "Env overrides:
 *      (none)"). *)

let order_source : string option ref = ref None
let injection_log : string list ref = ref []
let log_injection (s : string) = injection_log := s :: !injection_log

let resolve_order_source () : string option =
  match !order_source with
  | Some _ as s -> s
  | None -> Knobs.sched_order ()
;;

(* Canonical DAG description: one "tag:pred pred ..." line per node,
 * tag-ascending, preds tag-ascending. Independent of schedule order,
 * so su's order and any legal reordering of the SAME dag share a
 * signature, while any algsimp/CSE change that re-tags the dag does
 * not. *)
let dag_signature (nodes : (int * int list) list) : string =
  let canon =
    List.sort compare (List.map (fun (t, ps) -> t, List.sort compare ps) nodes)
  in
  let b = Buffer.create 4096 in
  List.iter
    (fun (t, ps) ->
       Buffer.add_string b (string_of_int t);
       Buffer.add_char b ':';
       List.iter
         (fun p ->
            Buffer.add_char b ' ';
            Buffer.add_string b (string_of_int p))
         ps;
       Buffer.add_char b '\n')
    canon;
  Digest.to_hex (Digest.string (Buffer.contents b))
;;

(* Read an order file: (tags, dagsig-if-present). '#'-prefixed lines
 * are metadata; non-integer lines are skipped, matching the tolerance
 * of the pre-existing readers (which already ignored them via
 * int_of_string_opt). *)
let read_order_file (f : string) : int list * string option =
  let ic = open_in f in
  let tags = ref [] in
  let dsig = ref None in
  (try
     while true do
       let line = String.trim (input_line ic) in
       if String.length line > 0 && line.[0] = '#'
       then (
         match String.split_on_char ' ' line with
         | [ "#dagsig"; s ] -> dsig := Some s
         | _ -> ())
       else (
         match int_of_string_opt line with
         | Some t -> tags := t :: !tags
         | None -> ())
     done
   with
   | End_of_file -> ());
  close_in ic;
  List.rev !tags, !dsig
;;

(* SU number — third comparator key, and the record of its trial.
 *
 * Classical Sethi-Ullman (1970) labels are EXACT minimum-register
 * counts for TREES. On a DAG the true generalization is Touati's
 * REGISTER SUFFICIENCY (min registers over all schedules of the
 * node's cone) — NP-complete on DAGs (Sethi 1975), so every occupant
 * of this slot is a proxy. The obvious DAG corrections were built and
 * raced (tools/ablate2.py, exact picker replica, 245/245 order match):
 *
 *   first-owner DAG-SU (one parent pays full, others see 1)
 *   shared-as-1 DAG-SU (all parents see cost 1 for multi-use values)
 *   dynamic kills (operands the node retires)
 *      -> ALL THREE SCHEDULE-IDENTICAL to classic SU at every prime.
 *
 * WHY the slot is saturated for that whole family: it only fires at
 * cp_dist ties, and cp-tied ready nodes on butterfly DAGs are
 * isomorphic same-layer siblings — their PRED-cones are congruent, so
 * any INPUT-side label assigns them equal values and falls to tag.
 * Only OUTPUT-side measures (properties of the USER cone) can
 * discriminate, because isomorphic siblings reach DIFFERENT sink
 * subsets. Two such measures shipped as VFFT_SU_TIEBREAK (env knob,
 * default = classic, byte-identical):
 *
 *   =cone      static: fewer reachable sinks first (popcount of the
 *              sink mask) — "commit to few outputs."
 *   =affinity  dynamic: larger sink-mask OVERLAP WITH THE LAST
 *              SCHEDULED NODE first — "stay in the committed cone,"
 *              SR's own philosophy formalized output-side.
 *
 * Race (gcc 13.3 insns/spills; win rule = fewer insns, spills <=
 * baseline, FMA invariant; all bit-exact where promoted):
 *
 *   R   classic      cone         AFFINITY        path
 *   4   55/0         55/0         55/0     inert  monolithic (order
 *                                                 position-identical)
 *   8   132/6        132/6¹       135/8    LOSES  monolithic
 *   11  317/42       314/39       306/41   win    monolithic
 *   13  446/70       424/55       417/59   win    monolithic
 *   17  756/175      762/176      741/169  win    monolithic (1st ever)
 *   19  876/192      866/182      868/189  win    monolithic
 *   23  1275/294     1282/311     1266/299 spills+5  monolithic
 *   25  1000/171     1007/185     1025/187 LOSES  blocked CT(5,5)
 *   32  1031/183     1027/183     1022/181 win    blocked
 *   64  2543/594     2548/585²    2579/610 LOSES  blocked
 *   ¹ cone@8 moves 10/69 positions; gcc lands on identical asm.
 *   ² cone@64 trades -9 spills for +5 insns: fails the insn gate.
 *
 * Affinity beats classic at 4/5 primes plus blocked R=32, and is
 * best-in-class on insns at all five primes — but LOSES on blocked
 * R=25/R=64. The mechanism is doc-70/71's own finding one level
 * down: on the blocked path the CONSTRUCTION already owns locality
 * (clusters are the cones), sinks-per-cluster are few, so the
 * output-side mask carries little information and mostly perturbs a
 * good source order. Where blocking does the shaping, the tie-break
 * is marginal — consistent with doc 71 closing blocked-comparator
 * work as low-EV. Note the annealer still beats affinity on pow2
 * (R=32 wisdom 996/168 vs 1022/181): on blocked codelets, SEARCH
 * remains the lever; affinity is the zero-cost deterministic option.
 * RUNTIME PROXIES (no direct runtime yet — container absolute
 * timing is 22% bimodal and these deltas sit below the Windows
 * host's paired-bench noise, the exact regime the static rule was
 * adopted for; the i9 paired A/B, MDE 0.1-0.3%, is the shipping
 * gate). Evidence stack: (1) realized-spill law, Spearman ~0.94 on
 * the i9 (Phase 0); (2) win-rule construction — insns down, spills
 * gated, FMA invariant, ordering-ILP bounded 1-3% (doc 67); (3)
 * llvm-mca 18 (alderlake model, main-loop block, 10 iter) on the
 * realized asm: R=11 wash (729 vs 730 cy), R=13 -1.7% (995->978),
 * R=17 -2.8% (1884->1832), R=19 +1.5% (2259->2293) — mca CONFIRMS
 * 13/17, and at R=19 BOTH models (Belady traffic 70->74 and mca)
 * mildly oppose the better statics: the R=19 wisdom entry is
 * PROVISIONAL, i9 arbitrates, drop if it loses there.
 *
 * The OPERATING WINDOW this table draws: affinity pays off exactly
 * on MID-SIZE MONOLITHIC DAGs — large enough that the register wall
 * bites hard (R=11..19: 42-192 baseline spills) so discovered cone
 * structure matters, but still monolithic so there IS structure to
 * discover. Below the wall (R=4: zero ties fire; R=8: 6 spills,
 * order barely matters and affinity's greed costs +2) the slot is
 * inert-to-harmful; above, on blocked codelets, the construction
 * already owns the cones. Tie-breaks refine what construction
 * leaves undecided — no more, no less.
 * VERDICT: classic stays the default; per-codelet A4 deployment
 * table as measured: affinity for {11,13,17,19,32}, classic for
 * {4,8,23,25,64} — i9 runtime gate pending before any promotion. Open interaction: doc-65 duplication selects targets
 * from emitted spans — affinity changes the spans, so dup-on-affinity
 * is an unmeasured stacking cell. The principled summary: the most
 * principled replacement for a tree label on a DAG is not a better
 * register formula (NP-complete target, input-side provably blind
 * here) but an OUTPUT-SIDE key raced per codelet — affinity being its
 * strongest measured instantiation. *)

(* Latency of producing this node, given a Uarch profile. Used by
 * critical-path computation. *)
let node_latency (uarch : Uarch.t) (e : Ir.t) : int =
  match e.node with
  | NK_Const _ -> 0 (* set1 broadcast: free / inlined *)
  | NK_Load _ -> uarch.load_l1_latency
  | NK_Neg _ -> uarch.add_latency (* xor with sign mask, ~ add latency *)
  | NK_Add _ | NK_Sub _ -> uarch.add_latency
  | NK_Mul _ -> uarch.mul_latency
  | NK_CmulRe _ | NK_CmulIm _ ->
    uarch.fma_latency
    (* Cmul emits as 1 mul + 1 fma;
     * latency dominated by fma_latency *)
  | NK_Fma _ -> uarch.fma_latency (* single FMA instruction *)
  | NK_Plus _ -> Ir.nk_plus_unreachable "schedule.ml:255"
;;

(* ═══════════════════════════════════════════════════════════════
 *  SCHEDULER NODE INTERFACE (zil_pipeline_port.md §11 — full-IL)
 *
 * The SR scheduler below is dependency-driven: its whole loop reads
 * `preds` and `tag`, and only FIVE places ever look at a node's payload
 * (latency, SU label, the STARVE load gate, the const leaf policy, and
 * the cosmetic dump char). That makes it reusable by ANY DAG, not just
 * the real-valued Ir — which is what the interleaved-complex (full-IL)
 * backend needs: it keeps its own packed-complex IR (merging into
 * Ir.node_kind would cost ~150 exhaustive-match arms across 7 modules,
 * most of them in real-valued passes an IL kernel never runs) and
 * borrows ONLY the scheduler, via this signature.
 *
 * The record shape `{ tag; node }` is part of the signature on purpose:
 * it keeps every `e.tag` in the body working unchanged, so functorizing
 * is a type-level move, not a rewrite of the algorithm.
 * ═══════════════════════════════════════════════════════════════ *)
module type SCHED_NODE = sig
  type payload

  type t =
    { tag : int
    ; node : payload
    }

  (* Immediate operands — the ONLY dependency accessor the loop uses. *)
  val preds : t -> t list

  (* Cycle cost of producing this node (drives cp_dist). *)
  val latency : Uarch.t -> t -> int

  (* STARVE rule: loads are admitted late, never competing with arith. *)
  val is_load : t -> bool

  (* B2 (memory ops as first-class scheduled nodes): STORE pseudo-nodes —
   * a store retires its pred value to memory and produces nothing. The
   * real-valued Ir has no store nodes (stores live in `assigns`;
   * su_schedule appends them), so both in-tree instantiations answer
   * `false` everywhere — the accessor is ADDITIVE and provably inert for
   * them. The zsplit combined-graph instantiation (Cascade_z.ZNode)
   * answers true for its ZStore units. The SR loop itself needs no store
   * rule — an in-graph store is already sink-classed by its empty user
   * set, so RETIRE fires it at readiness — but this accessor is the
   * contract by which policy/emission code and the B3 placement search
   * tell memory retirement apart from arithmetic sinks. *)
  val is_store : t -> bool

  (* Leaf policy under `Lookahead load admission. *)
  val is_const : t -> bool

  (* Cosmetic, VFFT_SCHED_DUMP sidecar only. *)
  val kind_char : t -> char
end

(* The real-valued Ir instantiation — restores every existing entry point
 * unchanged via `include Make (Ir_node)` below. *)
module Ir_node : SCHED_NODE with type payload = Ir.node_kind and type t = Ir.t = struct
  type payload = Ir.node_kind

  type t = Ir.t =
    { tag : int
    ; node : payload
    }

  let preds = Ir.preds
  let latency = node_latency

  let is_load (n : t) =
    match n.node with
    | NK_Load _ -> true
    | _ -> false
  ;;

  (* No store node kind in the real-valued Ir: stores are the assigns
   * list, appended by su_schedule (see SCHED_NODE's is_store card). *)
  let is_store (_ : t) = false

  let is_const (n : t) =
    match n.node with
    | NK_Const _ -> true
    | _ -> false
  ;;

  let kind_char (n : t) =
    match n.node with
    | NK_Const _ -> 'C'
    | NK_Load _ -> 'L'
    | NK_Add _ | NK_Sub _ -> 'A'
    | NK_Neg _ -> 'N'
    | NK_Mul _ -> 'M'
    | NK_Fma _ -> 'F'
    | NK_CmulRe _ | NK_CmulIm _ -> 'X'
    | NK_Plus _ -> 'O'
  ;;
end

module Make (N : SCHED_NODE) = struct
  open N

  (* Compute critical-path distance from each node to a sink, in cycles.
   * cp_dist[n] = latency(n) + max(cp_dist[user] for user in users(n))
   * Sinks have cp_dist = latency(n). *)
  let compute_cp_dist (uarch : Uarch.t) (sinks : t list) (all_nodes : t list)
    : (int, int) Hashtbl.t
    =
    (* Build successor (user) map. *)
    let users : (int, t list) Hashtbl.t = Hashtbl.create 256 in
    let add_user prod use =
      let cur =
        try Hashtbl.find users prod.tag with
        | Not_found -> []
      in
      Hashtbl.replace users prod.tag (use :: cur)
    in
    List.iter (fun n -> List.iter (fun p -> add_user p n) (preds n)) all_nodes;
    let cp_dist : (int, int) Hashtbl.t = Hashtbl.create 256 in
    let sink_tags =
      List.fold_left
        (fun acc s ->
           Hashtbl.replace acc s.tag ();
           acc)
        (Hashtbl.create 16)
        sinks
    in
    (* Process in tag-descending order. Tag-ascending = topological (children
     * before parents in our hash-cons), so tag-descending = reverse-topological,
     * which is what we want for backward DP. *)
    let sorted_desc = List.sort (fun a b -> compare b.tag a.tag) all_nodes in
    List.iter
      (fun n ->
         let lat = latency uarch n in
         let user_list =
           try Hashtbl.find users n.tag with
           | Not_found -> []
         in
         let max_user_cp =
           List.fold_left
             (fun acc u ->
                let u_cp =
                  try Hashtbl.find cp_dist u.tag with
                  | Not_found -> 0
                in
                max acc u_cp)
             0
             user_list
         in
         let cp = if Hashtbl.mem sink_tags n.tag then lat else lat + max_user_cp in
         Hashtbl.replace cp_dist n.tag cp)
      sorted_desc;
    cp_dist
  ;;

  (* Compute SU number — register-pressure approximation.
   * For trees, classical SU labels are exact. On DAGs (shared subexprs),
   * this over-estimates pressure for shared values, which is a
   * conservative bias toward scheduling them earlier. *)
  (* PAYLOAD-GENERIC form. The k-ary label — sort children's SU
     descending, label = max_i (su_i + i) — reproduces every per-kind
     case the hand-written version spelled out, so this is a refactor,
     not a behavior change:
       leaf (no preds)          -> 1                      (special-cased)
       unary [a]                -> max(sa+0)      = sa
       binary [a;b], sa>=sb     -> max(sa, sb+1)
             ... which equals the old `if sa=sb then sa+1 else max sa sb`
             (sa=sb: max(v,v+1)=v+1; sa>sb: sb+1<=sa so it is sa)
       3-ary / 4-ary            -> the same formula the old code already
                                   used verbatim for Fma / Cmul.
     Verified empirically too: with this in place every production codelet
     regenerates BYTE-IDENTICAL (the R0 baseline diff). *)
  let compute_su_number (all_nodes : t list) : (int, int) Hashtbl.t =
    let su : (int, int) Hashtbl.t = Hashtbl.create 256 in
    let get_su (e : t) : int =
      try Hashtbl.find su e.tag with
      | Not_found -> 1
    in
    let sorted_asc = List.sort (fun a b -> compare a.tag b.tag) all_nodes in
    List.iter
      (fun n ->
         let s =
           match List.map get_su (preds n) with
           | [] -> 1
           | sus ->
             let sus = List.sort (fun x y -> compare y x) sus in
             let rec compute idx = function
               | [] -> 0
               | s :: rest -> max (s + idx) (compute (idx + 1) rest)
             in
             compute 0 sus
         in
         Hashtbl.add su n.tag s)
      sorted_asc;
    su
  ;;

  (* ============================================================
   * STARVE–RETIRE (SR) LIST SCHEDULING — the production scheduler.
   * (Historical flag name: --su. The Sethi–Ullman label survives only
   * as a near-neutral tie-break; the algorithm is bespoke. Named after
   * the two rules the ablation proved load-bearing — see below.)
   *
   * THE TWO RULES
   *   STARVE — starvation-gated load admission (late birth).
   *     Loads never compete with arithmetic. A load is admitted only
   *     when the arithmetic ready-set is EMPTY, and then in source
   *     order. A value is born at the last moment demand forces it.
   *   RETIRE — eager sink retirement (early death).
   *     A ready sink (store root / node with no users) always wins.
   *     Values die at the first legal moment, freeing their registers.
   *   Together: live ranges are compressed from both ends. Intuition
   *   (open conjecture, doc 69 §4): this is the scheduling-side DUAL of
   *   Belady's MIN eviction — Belady evicts the value whose next use is
   *   farthest; SR constructs the order so values are born as late and
   *   die as early as the DAG permits.
   *
   * PRIORITY (lexicographic, arithmetic ready-set):
   *   1. sink-first            — RETIRE
   *   2. cp_dist DESC          — tie-break; ablates to ±2, radix-dep sign
   *   3. su_num ASC            — tie-break; "the reach"; ablates to ~0
   *   4. tag ASC               — determinism
   *   Loads: only on starvation, source order (order among loads is a
   *   no-op given laziness — measured).
   *
   * ABLATION LEDGER (tools/ablate.py: exact replica of this picker,
   * validated 245/245 order match at every radix, then component
   * knockout; model = Belady traffic @16 regs, tools/traffic.py; R=13):
   *   production (all keys)        traffic 30   gcc 446/70
   *   sink-first + lazy loads ONLY traffic 28   gcc 431/68
   *   tag-only + lazy loads        traffic 43   gcc 431/68
   *   load law REMOVED             traffic 230  gcc 502/132  <- ~8x
   * The load law is the algorithm; sink-first is the second factor;
   * everything else is noise kept for determinism and history.
   *
   * CERTIFICATION (docs 67/68/69): at R=13 the SR order's Belady
   * traffic (28–30) sits ~3x below the best of 200,000 random legal
   * orders (87); a 4,000-move hill climb cannot improve it; three
   * literature-grounded successors (Belady-greedy, MRIS lineage
   * sequencing, beam-Belady) raced and lost IN-MODEL (110/98/90).
   * Under the complete objective — cycles on M(R,W,ports,latencies)
   * with Belady-optimal spilling — pressure-first is the operating
   * point for every W >= ~32 (doc 68 sigma*(R,W)); issue-oriented
   * ordering is worth ~1–3% on >=32-entry-window hardware (doc 67) and
   * is deliberately not chased.
   *
   * The remaining deterministic improvement lives in the DAG, not the
   * order: the selective-duplication pass (doc 65). Compiler-specific
   * residue (rematerialization/folding) is measured as a tax, never
   * chased with per-compiler scheduling.
   * ============================================================ *)

  (* ============================================================
   * EXPERIMENTAL RECORD — what was raced against this scheduler,
   * what was measured, and where the full evidence lives.
   * (Container campaign, 2026-07-02, gcc 13.3 / clang 18 /
   * raptorlake; magnitudes are toolchain-specific, mechanisms are
   * not. Numbers are insns/spills for gcc unless stated. Docs are
   * in src/dag-fft-compiler/docs/ unless pathed.)
   *
   * 1. RIVAL SCHEDULERS, RACED AND LOST (doc 69; tools/ keeps them
   *    as recorded negatives). Model = Belady traffic @16 regs,
   *    R=13; SR scores 30:
   *      Belady-in-the-loop greedy (tools/lineage_sched.py v1) 110
   *      MRIS lineage-continuation (v2)                     98-class
   *      beam-Belady, width 4..64 (tools/beam_sched.py)          90
   *    Root cause of all three: traffic is a CLIFF function (zero
   *    gradient until the file overflows), so one-step keys decide
   *    the damage phase blind; SR's implicit cone traversal never
   *    sees the cliff.
   *
   * 2. FLOORS (docs 67/69): 200,000 random legal orders bottom out
   *    at traffic 87; a 4,000-move hill climb from SR cannot leave
   *    30. In-order-issue exhaustive B&B closed R=3 EXACTLY (18);
   *    even at N=20 there are >2e8 linear extensions, so B&B with
   *    admissible bounds IS the exhaustive method.
   *
   * 3. LEAF-POLICY KNOBS, ALL REGRESSED (kept default-off below as
   *    VFFT_SCHED_LOADS): anyorder / lookahead(p2) / nearstarve
   *    took R=13 from 70 to 114-115 spills. The annealed winner's
   *    "loads travel earlier" pattern is a property of a jointly
   *    optimized solution, not a decision rule. Source order AMONG
   *    loads is a no-op given laziness (ablation: any == src).
   *
   * 4. GH THRESHOLD SWEEP: flat at every value 4..12 (phase-0 doc)
   *    — peak_live is the wrong pressure quantity; TRAFFIC is the
   *    right one (doc 68 F2: anyorder has LOWER MAXLIVE than SR,
   *    30 vs 32, and 3.5x the Belady traffic, 114 vs 32).
   *
   * 5. ISSUE-OPTIMAL ORDERING LOSES WITH REGISTERS IN THE OBJECTIVE
   *    (docs 67/68): the min-in-order order (model 148 vs SR 227)
   *    costs 124 gcc spills / 140 znver3 / 167 clang and 248 Belady
   *    ops — worst-in-class everywhere that has 16 registers. And
   *    hardware refunds SR's in-order deficit: by window 32-64 even
   *    the adversarially WORST order lands within a few % of the
   *    dataflow floor; at Raptor Lake's ~200-entry scheduler,
   *    static ordering's in-core ILP contribution is ~1-3%.
   *
   * 6. THE ANNEALER (doc 66; tools/anneal_linux.py,
   *    blocked_anneal.py; wisdom in generator/sched_wisdom/,
   *    consumed via VFFT_SCHED_WISDOM): R=13 70->51; R=32 blocked
   *    183->168; R=64 blocked 594->585 (pinned = permutation
   *    exhausted, which motivated duplication). Post-duplication
   *    role: retired on primes, sole working lever on blocked pow2,
   *    permanent ceiling instrument + toolchain-transfer tool.
   *
   * 7. MINIMAX ACROSS COMPILERS (doc 68 §5; tools/robust_anneal.py):
   *    a single order at gcc 392/50 AND clang 540/114 — at/below
   *    every per-target floor simultaneously (clang's own dedicated
   *    260-iter search only reached 131; clang is order-insensitive:
   *    misched-off makes things WORSE, allocator sweep greedy 136 /
   *    pbqp 144 / basic 170 / fast 273). Shipped as the R=13 wisdom
   *    entry. Residual clang-vs-gcc = allocator tax, not
   *    order-payable; confined to 16-register targets (znver4-class
   *    EVEX files collapse spills to 27/28/5).
   *
   * 8. DUPLICATION (doc 65) — input-level, not this file: cloning
   *    long-span leaf-fed values beats every ordering result on
   *    primes (R=13: 377/31 from PLAIN SR order, better than the
   *    converged search) and is negative on pow2 in all variants.
   *    The pending OCaml algsimp pass (v5 coverage selector + v4
   *    chained mode, doc 65 §8) is the one remaining deterministic
   *    improvement; it changes the DAG this scheduler orders.
   *
   * 9. SELF-SPILLING FALSIFIED / BLOCKING VINDICATED (docs 70/71):
   *    generator-owned Belady-optimal explicit spilling loses to
   *    the blocked construction IN-PLAN (76-88 vs 52-59 memory ops
   *    at R=16), and compilers erase explicit spill arrays anyway —
   *    including the doc-58 seam itself, which never exists in the
   *    assembly (fully scalarized, both compilers). Blocking wins
   *    by SHAPING: it lowers this DAG's Belady order-floor 3.3-4.7x
   *    (70->21 at R=16, 249->53 at R=32). The realized-vs-floor gap
   *    (+38/+130 under gcc) is allocator tax, bounded ~10%
   *    order-payable by the annealer's direct search — so the
   *    subset comparator below (port-balance / GH keys) stays
   *    UN-ABLATED as accepted, documented debt (doc 71 §4).
   *
   * 10. CONVENTION NOTE: dump-based traffic excludes store sinks as
   *     users (outputs cost nothing); tools/c_traffic.py includes
   *     them. All RELATIVE results above used one convention
   *     throughout; absolute traffic numbers are convention-tagged.
   *
   * 11. TIE-BREAK SLOT RESOLVED (see compute_su_number's header for
   *     the full table): every INPUT-side label (classic SU, two DAG-SU
   *     corrections, kills) is schedule-identical — cp-tied ready nodes
   *     are isomorphic siblings, so pred-cone labels are provably blind
   *     here; only OUTPUT-side keys discriminate. VFFT_SU_TIEBREAK=
   *     affinity (sink-mask overlap with last scheduled) beats classic
   *     at 4/5 primes incl. the never-before-improved R=17 (756/175 ->
   *     741/169), best insns at all five; R=23 spill gate fails by +5.
   *     BLOCKED-PATH EXTENSION (user-priority R=25/32/64): knob wired
   *     into su_schedule_subset (subset-relative masks, per-cluster
   *     last-mask). Affinity wins blocked R=32 (1031/183 -> 1022/181,
   *     bit-exact) but LOSES R=25 (+16 spills) and R=64 (+16 insns) —
   *     construction owns locality on the blocked path, so the slot is
   *     marginal there (doc 71 concurs); annealer wisdom still beats
   *     affinity on pow2 (996/168 at R=32). Classic remains default;
   *     A4 table: affinity {11,13,17,19,32}, classic {23,25,64}.
   *     PER-CODELET DEPLOYMENT: resolved through the EXISTING wisdom
   *     mechanism, not a whitelist — affinity's novel winners ship as
   *     sched_wisdom entries (radix{11,17,19}, provenance-commented,
   *     dagsig'd, bit-exact); R=13 keeps minimax (392/50 < 417/59),
   *     R=32 keeps annealed (996/168 < 1022/181). Wisdom dir = single
   *     per-codelet best-known table across {search, minimax,
   *     affinity}. i9 runtime gate pending; dup-on-affinity spans
   *     unmeasured.
   *
   * Open, honestly: WHY starvation-gated load admission + eager
   * sink retirement minimizes Belady traffic on butterfly DAGs —
   * the theorem this record points at (doc 69 §4).
   * ============================================================ *)
  let su_schedule (uarch : Uarch.t) (assigns : (Expr.elem_ref * t) list)
    : (Expr.elem_ref option * t) list
    =
    (* Collect all reachable nodes. *)
    let seen : (int, t) Hashtbl.t = Hashtbl.create 256 in
    let rec visit (e : t) =
      if not (Hashtbl.mem seen e.tag)
      then (
        Hashtbl.add seen e.tag e;
        List.iter visit (preds e))
    in
    List.iter (fun (_, e) -> visit e) assigns;
    let all_nodes = Hashtbl.fold (fun _ n acc -> n :: acc) seen [] in
    let sinks = List.map snd assigns in
    let cp_dist = compute_cp_dist uarch sinks all_nodes in
    let su_num = compute_su_number all_nodes in
    (* Successor map for forward propagation when a node becomes scheduled. *)
    let users : (int, t list) Hashtbl.t = Hashtbl.create 256 in
    List.iter
      (fun n ->
         List.iter
           (fun p ->
              let cur =
                try Hashtbl.find users p.tag with
                | Not_found -> []
              in
              Hashtbl.replace users p.tag (n :: cur))
           (preds n))
      all_nodes;
    (* Unscheduled-pred counter per node. *)
    let unsched_count : (int, int) Hashtbl.t = Hashtbl.create 256 in
    List.iter (fun n -> Hashtbl.add unsched_count n.tag (List.length (preds n))) all_nodes;
    (* Ready set: nodes whose predecessors have all been scheduled. *)
    let scheduled : (int, unit) Hashtbl.t = Hashtbl.create 256 in
    let in_ready : (int, unit) Hashtbl.t = Hashtbl.create 64 in
    let ready : t list ref = ref [] in
    List.iter
      (fun n ->
         if preds n = []
         then (
           ready := n :: !ready;
           Hashtbl.add in_ready n.tag ()))
      all_nodes;
    (* Load source-order tracking: only allow the next-required load to fire. *)
    let load_tags =
      List.filter_map (fun n -> if is_load n then Some n.tag else None) all_nodes
      |> List.sort compare
    in
    let load_array = Array.of_list load_tags in
    let load_idx = ref 0 in
    let next_required_load () : int option =
      if !load_idx < Array.length load_array then Some load_array.(!load_idx) else None
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
    let is_sink n =
      let user_list =
        try Hashtbl.find users n.tag with
        | Not_found -> []
      in
      user_list = []
    in
    (* === LEAF-PLACEMENT POLICY (env-gated; default = unchanged) ===
     * Annealer forensics on R=13 (docs/performance/schedule_search_phase2):
     * the search's entire win was LEAF placement, not arithmetic order —
     * loads moved EARLIER (first-use distance median 1 -> 8, source order
     * broken in 88/325 pairs) and constants moved LATER (55 -> 21). The
     * strict policy below is therefore a measured blind spot, in two parts:
     * loads are maximally lazy (source-order, fire only on starvation), and
     * NK_Const is NOT a load, so constants compete as arithmetic where
     * their high cp_dist fires them absurdly early.
     *
     * VFFT_SCHED_LOADS=
     *   unset/strict : the original policy, byte-identical output.
     *   anyorder     : starvation-only firing kept, but the comparator
     *                  picks among ALL ready loads (source-order law
     *                  dropped).
     *   lookahead    : loads AND NK_Const are deferrable leaves; prefer
     *                  arithmetic, but after every VFFT_LOAD_PACE (default
     *                  4) arithmetic picks, fire the best HOT leaf — one
     *                  that is the last missing operand of some consumer —
     *                  giving leaves lead time without front-loading.
     *                  Starvation fires the best ready leaf regardless.
     *   nearstarve   : fire the best hot LOAD when |arith_ready| <=
     *                  VFFT_LOAD_PACE (just before starvation).
     *
     * RACE VERDICT (container, gcc 13.3, R=13/17/32/64): all three
     * non-strict modes REGRESS — anyorder R=13 70->115 spills, lookahead
     * and nearstarve similar or worse; nearstarve T=1 exactly reproduces
     * anyorder, isolating the damage to the load CHOOSER, not the firing
     * time. The source-order law is protective under greedy selection.
     * The leaf-placement headroom is real (annealer: -27% spills at R=13,
     * first-use distance 1->8, order broken 88/325 pairs) but is reachable
     * only by global search, not by these greedy rules. Knobs retained,
     * default strict, so the ledger shows the race. *)
    let load_policy =
      match Knobs.sched_loads () with
      | Some "anyorder" -> `Anyorder
      | Some "lookahead" -> `Lookahead
      | Some "nearstarve" -> `Nearstarve
      | _ -> `Strict
    in
    let leaf_pace =
      match Knobs.load_pace () with
      | Some s ->
        (try max 1 (int_of_string s) with
         | _ -> 4)
      | None -> 4
    in
    let since_leaf = ref 0 in
    let is_leaf n = is_load n || (load_policy = `Lookahead && is_const n) in
    let is_hot n =
      List.exists
        (fun (u : t) ->
           (try Hashtbl.find unsched_count u.tag with
            | Not_found -> 0)
           = 1)
        (try Hashtbl.find users n.tag with
         | Not_found -> [])
    in
    (* VFFT_SU_TIEBREAK = cone | affinity (default classic, byte-
     * identical). Full theory, race table, and verdict: see the SU
     * NUMBER comment above compute_su_number. Sink bitmask capped at
     * 63; larger DAGs fall back to classic. *)
    let tiebreak =
      match Knobs.su_tiebreak () with
      | Some "cone" -> `Cone
      | Some "affinity" -> `Affinity
      | _ -> `Classic
    in
    let cone_mode = tiebreak <> `Classic in
    let cone_count : (int, int) Hashtbl.t = Hashtbl.create 256 in
    let sink_mask : (int, Int64.t) Hashtbl.t = Hashtbl.create 256 in
    let last_mask : Int64.t ref = ref 0L in
    if cone_mode && List.length (List.filter (fun (n : t) -> is_sink n) all_nodes) <= 63
    then (
      let sink_idx : (int, int) Hashtbl.t = Hashtbl.create 64 in
      let next = ref 0 in
      List.iter
        (fun (n : t) ->
           if is_sink n
           then (
             Hashtbl.add sink_idx n.tag !next;
             incr next))
        all_nodes;
      let masks : (int, Int64.t) Hashtbl.t = Hashtbl.create 256 in
      let rec mask_of (n : t) : Int64.t =
        match Hashtbl.find_opt masks n.tag with
        | Some m -> m
        | None ->
          let m =
            match Hashtbl.find_opt users n.tag with
            | None | Some [] -> Int64.shift_left 1L (Hashtbl.find sink_idx n.tag)
            | Some us -> List.fold_left (fun acc u -> Int64.logor acc (mask_of u)) 0L us
          in
          Hashtbl.add masks n.tag m;
          m
      in
      let popcount m =
        let c = ref 0
        and x = ref m in
        while !x <> 0L do
          c := !c + Int64.to_int (Int64.logand !x 1L);
          x := Int64.shift_right_logical !x 1
        done;
        !c
      in
      List.iter
        (fun (n : t) ->
           let m = mask_of n in
           Hashtbl.replace sink_mask n.tag m;
           Hashtbl.replace cone_count n.tag (popcount m))
        all_nodes);
    let pc64 (m : Int64.t) : int =
      let c = ref 0
      and x = ref m in
      while !x <> 0L do
        c := !c + Int64.to_int (Int64.logand !x 1L);
        x := Int64.shift_right_logical !x 1
      done;
      !c
    in
    let pick_next () : t option =
      if !ready = []
      then None
      else (
        let cmp a b =
          let sa = is_sink a in
          let sb = is_sink b in
          if sa <> sb
          then compare sb sa
          else (
            let cpa =
              try Hashtbl.find cp_dist a.tag with
              | Not_found -> 0
            in
            let cpb =
              try Hashtbl.find cp_dist b.tag with
              | Not_found -> 0
            in
            if cpa <> cpb
            then compare cpb cpa (* higher cp_dist first *)
            else if tiebreak = `Affinity && Hashtbl.length sink_mask > 0
            then (
              let ov t =
                pc64
                  (Int64.logand
                     !last_mask
                     (try Hashtbl.find sink_mask t with
                      | Not_found -> 0L))
              in
              let oa = ov a.tag
              and ob = ov b.tag in
              if oa <> ob
              then compare ob oa (* larger overlap with last first *)
              else compare a.tag b.tag)
            else if tiebreak = `Cone && Hashtbl.length cone_count > 0
            then (
              let ca =
                try Hashtbl.find cone_count a.tag with
                | Not_found -> 0
              in
              let cb =
                try Hashtbl.find cone_count b.tag with
                | Not_found -> 0
              in
              if ca <> cb
              then compare ca cb (* fewer reachable sinks first *)
              else compare a.tag b.tag)
            else (
              let sua =
                try Hashtbl.find su_num a.tag with
                | Not_found -> 0
              in
              let sub =
                try Hashtbl.find su_num b.tag with
                | Not_found -> 0
              in
              if sua <> sub
              then compare sua sub (* lower su_num breaks ties *)
              else compare a.tag b.tag (* stable: tag-ascending *)))
        in
        match load_policy with
        | `Strict ->
          let arith_ready = List.filter (fun n -> not (is_load n)) !ready in
          (match arith_ready with
           | _ :: _ ->
             (* At least one arithmetic op ready — pick the best one, defer loads. *)
             Some (List.hd (List.sort cmp arith_ready))
           | [] ->
             (* No arithmetic ready — must fire a load to unblock more work.
              * Fire the next required load (preserves source order). *)
             let req_load = next_required_load () in
             let load_candidates =
               List.filter (fun n -> is_load n && Some n.tag = req_load) !ready
             in
             (match load_candidates with
              | [] -> None (* shouldn't happen if scheduling is making progress *)
              | _ -> Some (List.hd load_candidates)))
        | `Anyorder ->
          let arith_ready = List.filter (fun n -> not (is_load n)) !ready in
          (match arith_ready with
           | _ :: _ -> Some (List.hd (List.sort cmp arith_ready))
           | [] ->
             (match List.filter is_load !ready with
              | [] -> None
              | ls -> Some (List.hd (List.sort cmp ls))))
        | `Lookahead ->
          let leaves, arith = List.partition is_leaf !ready in
          let hot_leaves = List.filter is_hot leaves in
          if !since_leaf >= leaf_pace && hot_leaves <> []
          then (
            since_leaf := 0;
            Some (List.hd (List.sort cmp hot_leaves)))
          else (
            match arith with
            | _ :: _ ->
              incr since_leaf;
              Some (List.hd (List.sort cmp arith))
            | [] ->
              since_leaf := 0;
              (match leaves with
               | [] -> None
               | ls -> Some (List.hd (List.sort cmp ls))))
        | `Nearstarve ->
          (* Fire the best HOT load when the arithmetic pool is nearly dry
           * (|arith_ready| <= VFFT_LOAD_PACE) — one to a few slots BEFORE
           * starvation would force it, instead of at starvation. Loads
           * only; constants keep their strict (arith-pool) behavior. *)
          let arith = List.filter (fun n -> not (is_load n)) !ready in
          let hot_loads = List.filter is_hot (List.filter is_load !ready) in
          if List.length arith <= leaf_pace && hot_loads <> []
          then Some (List.hd (List.sort cmp hot_loads))
          else (
            match arith with
            | _ :: _ -> Some (List.hd (List.sort cmp arith))
            | [] ->
              (match List.filter is_load !ready with
               | [] -> None
               | ls -> Some (List.hd (List.sort cmp ls)))))
    in
    let result : (Expr.elem_ref option * t) list ref = ref [] in
    let rec loop () =
      match pick_next () with
      | None -> ()
      | Some n ->
        Hashtbl.add scheduled n.tag ();
        ready := List.filter (fun x -> x.tag <> n.tag) !ready;
        Hashtbl.remove in_ready n.tag;
        if is_load n then incr load_idx;
        result := (None, n) :: !result;
        (if cone_mode
         then
           last_mask
           := try Hashtbl.find sink_mask n.tag with
              | Not_found -> !last_mask);
        let user_list =
          try Hashtbl.find users n.tag with
          | Not_found -> []
        in
        List.iter
          (fun u ->
             let cur =
               try Hashtbl.find unsched_count u.tag with
               | Not_found -> 0
             in
             let new_count = cur - 1 in
             Hashtbl.replace unsched_count u.tag new_count;
             if
               new_count = 0
               && (not (Hashtbl.mem in_ready u.tag))
               && not (Hashtbl.mem scheduled u.tag)
             then (
               ready := u :: !ready;
               Hashtbl.add in_ready u.tag ()))
          user_list;
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
     *   compile error, which the search treats as an invalid candidate.
     * Wisdom-grade additions (see SCHEDULE WISDOM PLUMBING above): the dump
     * carries a #dagsig header; injection resolves through order_source
     * (emit_c sets it from VFFT_SCHED_WISDOM/<codelet symbol>), verifies the
     * dagsig, checks completeness, and logs every accept/refuse. The order
     * source may name the file exactly or omit a ".txt" suffix (the wisdom
     * directory convention). *)
    let mono_sig () =
      dag_signature
        (List.map
           (fun (_, (n : t)) -> n.tag, List.map (fun (p : t) -> p.tag) (preds n))
           su_intermediates)
    in
    (match Knobs.Trace.sched_dump () with
     | None -> ()
     | Some file ->
       let oc = open_out file in
       Printf.fprintf oc "#dagsig %s\n" (mono_sig ());
       List.iter
         (fun (_, (n : t)) ->
            Printf.fprintf
              oc
              "%d:%s\n"
              n.tag
              (String.concat
                 " "
                 (List.map (fun (p : t) -> string_of_int p.tag) (preds n))))
         su_intermediates;
       close_out oc;
       (* Kinds sidecar for machine-model tooling (ILP-floor searcher):
        * one "tag KIND" line per node. C const, L load, A add/sub,
        * N neg, M mul, F fma, X cmul (emits as mul+fma pair: 2 uops on
        * the mul/fma port class). New file, same env gate; existing
        * dump consumers are unaffected. *)
       let kc = open_out (file ^ ".kinds") in
       List.iter
         (fun (_, (n : t)) ->
            let k = kind_char n in
            Printf.fprintf kc "%d %c\n" n.tag k)
         su_intermediates;
       close_out kc);
    let intermediates =
      match resolve_order_source () with
      | None -> su_intermediates
      | Some base ->
        let file =
          if Sys.file_exists base
          then Some base
          else if Sys.file_exists (base ^ ".txt")
          then Some (base ^ ".txt")
          else None
        in
        (match file with
         | None -> su_intermediates
         | Some f ->
           let tags, file_sig = read_order_file f in
           (match file_sig with
            | Some s when s <> mono_sig () ->
              log_injection
                (Printf.sprintf
                   "sched order STALE (dagsig mismatch), fell back to su: %s"
                   f);
              su_intermediates
            | _ ->
              let by_tag : (int, t) Hashtbl.t = Hashtbl.create 256 in
              List.iter (fun (n : t) -> Hashtbl.replace by_tag n.tag n) all_nodes;
              let ordered =
                List.filter_map
                  (fun t ->
                     match Hashtbl.find_opt by_tag t with
                     | Some n -> Some (None, n)
                     | None -> None)
                  tags
              in
              if List.length ordered <> List.length su_intermediates
              then (
                log_injection
                  (Printf.sprintf
                     "sched order INCOMPLETE (%d of %d nodes), fell back to su: %s"
                     (List.length ordered)
                     (List.length su_intermediates)
                     f);
                su_intermediates)
              else (
                log_injection
                  (Printf.sprintf
                     "sched order injected: %s (nodes=%d, %s)"
                     f
                     (List.length ordered)
                     (match file_sig with
                      | Some _ -> "dagsig verified"
                      | None -> "no dagsig, unverified"));
                ordered)))
    in
    let stores = List.map (fun (oref, e) -> Some oref, e) assigns in
    intermediates @ stores
  ;;

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
end

include Make (Ir_node)

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
let su_schedule_subset
      (uarch : Uarch.t)
      ~(gh : bool)
      ~(subset : Ir.t list)
      ~(sinks : Ir.t list)
  : Ir.t list
  =
  (* Subset membership lookup. *)
  let in_subset : (int, unit) Hashtbl.t = Hashtbl.create 256 in
  List.iter (fun n -> Hashtbl.add in_subset n.Ir.tag ()) subset;
  (* Sinks lookup for births() — sinks stay live even when remaining_users hits 0. *)
  let sink_set : (int, unit) Hashtbl.t = Hashtbl.create 64 in
  List.iter (fun s -> Hashtbl.replace sink_set s.Ir.tag ()) sinks;
  (* VFFT_SU_TIEBREAK = cone | affinity on the BLOCKED path (default
   * classic, byte-identical). Subset-relative: "sinks" here are the
   * caller-passed cluster sinks; masks/popcounts are computed within
   * this subset only, and affinity's last-scheduled memory resets per
   * cluster (each su_schedule_subset call is one contiguous emission
   * region). Same theory as the monolithic knob — see the SU NUMBER
   * header above compute_su_number. Cap 63 sinks per cluster. *)
  let tiebreak_sub =
    match Knobs.su_tiebreak () with
    | Some "cone" -> `Cone
    | Some "affinity" -> `Affinity
    | _ -> `Classic
  in
  let sub_mask : (int, Int64.t) Hashtbl.t = Hashtbl.create 256 in
  let sub_cone : (int, int) Hashtbl.t = Hashtbl.create 256 in
  let sub_last : Int64.t ref = ref 0L in
  let pc64s (m : Int64.t) : int =
    let c = ref 0
    and x = ref m in
    while !x <> 0L do
      c := !c + Int64.to_int (Int64.logand !x 1L);
      x := Int64.shift_right_logical !x 1
    done;
    !c
  in
  (* cp_dist over the full reachable graph from sinks. This includes
   * predecessors outside the subset (they contribute to depth).
   * We use compute_cp_dist's existing logic which walks transitively. *)
  let all_reachable : (int, Ir.t) Hashtbl.t = Hashtbl.create 512 in
  let rec visit (e : Ir.t) =
    if not (Hashtbl.mem all_reachable e.tag)
    then (
      Hashtbl.add all_reachable e.tag e;
      List.iter visit (preds e))
  in
  List.iter visit sinks;
  let all_nodes = Hashtbl.fold (fun _ n acc -> n :: acc) all_reachable [] in
  let cp_dist = compute_cp_dist uarch sinks all_nodes in
  let su_num = compute_su_number all_nodes in
  (* Successor map RESTRICTED to subset edges. *)
  let users : (int, Ir.t list) Hashtbl.t = Hashtbl.create 256 in
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
         (preds n))
    subset;
  if tiebreak_sub <> `Classic && List.length sinks <= 63
  then (
    let sink_idx : (int, int) Hashtbl.t = Hashtbl.create 64 in
    List.iteri (fun i (sk : Ir.t) -> Hashtbl.replace sink_idx sk.tag i) sinks;
    let rec mask_of (n : Ir.t) : Int64.t =
      match Hashtbl.find_opt sub_mask n.tag with
      | Some m -> m
      | None ->
        let own =
          match Hashtbl.find_opt sink_idx n.tag with
          | Some i -> Int64.shift_left 1L i
          | None -> 0L
        in
        let us =
          try Hashtbl.find users n.tag with
          | Not_found -> []
        in
        let m =
          List.fold_left (fun acc (u : Ir.t) -> Int64.logor acc (mask_of u)) own us
        in
        Hashtbl.add sub_mask n.tag m;
        m
    in
    List.iter
      (fun (n : Ir.t) -> Hashtbl.replace sub_cone n.tag (pc64s (mask_of n)))
      subset);
  (* Unscheduled-pred counter — only count predecessors IN the subset.
   * Out-of-subset preds are treated as already satisfied. *)
  let unsched_count : (int, int) Hashtbl.t = Hashtbl.create 256 in
  List.iter
    (fun n ->
       let in_subset_preds =
         List.filter (fun p -> Hashtbl.mem in_subset p.Ir.tag) (preds n)
       in
       Hashtbl.add unsched_count n.tag (List.length in_subset_preds))
    subset;
  (* Remaining-user counter (for Goodman-Hsu): how many in-subset successors
   * of node X are still unscheduled. Initially = total in-subset users.
   * Decremented when a user is scheduled. When hits 0, X is no longer live
   * (unless it's a sink). *)
  let remaining_users : (int, int) Hashtbl.t = Hashtbl.create 256 in
  if gh
  then (
    (* Pre-build successor map restricted to subset edges.
     * Same logic as the `users` table built below; we count entries here. *)
    let user_count : (int, int) Hashtbl.t = Hashtbl.create 256 in
    List.iter (fun n -> Hashtbl.add user_count n.Ir.tag 0) subset;
    List.iter
      (fun n ->
         List.iter
           (fun p ->
              if Hashtbl.mem in_subset p.Ir.tag
              then (
                let cur =
                  try Hashtbl.find user_count p.tag with
                  | Not_found -> 0
                in
                Hashtbl.replace user_count p.tag (cur + 1)))
           (preds n))
      subset;
    Hashtbl.iter (fun tag c -> Hashtbl.add remaining_users tag c) user_count);
  (* Live set tracker (for Goodman-Hsu only). Threshold uses the
   * uarch-specific pressure_threshold (24 for AVX-512, 12 for AVX2),
   * which already accounts for cmul/FMA scratch slack. *)
  let live : (int, unit) Hashtbl.t = Hashtbl.create 64 in
  let live_count () = Hashtbl.length live in
  (* VFFT_GH_THRESHOLD env override (experiment knob; default = unchanged).
   * Lowering it makes GH pressure-mode engage earlier -> frees registers
   * sooner -> fewer stack spills, at a possible latency cost. *)
  let threshold =
    match Knobs.gh_threshold () with
    | Some s ->
      (try int_of_string s with
       | _ -> uarch.Uarch.pressure_threshold)
    | None -> uarch.Uarch.pressure_threshold
  in
  (* Ready set: nodes with no unscheduled subset preds. *)
  let scheduled : (int, unit) Hashtbl.t = Hashtbl.create 256 in
  let in_ready : (int, unit) Hashtbl.t = Hashtbl.create 64 in
  let ready : Ir.t list ref = ref [] in
  List.iter
    (fun n ->
       if Hashtbl.find unsched_count n.tag = 0
       then (
         ready := n :: !ready;
         Hashtbl.add in_ready n.tag ()))
    subset;
  let is_load n =
    match n.Ir.node with
    | NK_Load _ -> true
    | _ -> false
  in
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
    match n.Ir.node with
    | NK_Mul _ | NK_CmulRe _ | NK_CmulIm _ | NK_Fma _ -> `P01
    | NK_Add _ | NK_Sub _ | NK_Neg _ -> `P5
    | _ -> `Other
  in
  let p01_fired = ref 0 in
  let p5_fired = ref 0 in
  (* Load source order — only relevant if subset contains loads. *)
  let load_tags =
    List.filter_map (fun n -> if is_load n then Some n.Ir.tag else None) subset
    |> List.sort compare
  in
  let load_array = Array.of_list load_tags in
  let load_idx = ref 0 in
  let next_required_load () =
    if !load_idx < Array.length load_array then Some load_array.(!load_idx) else None
  in
  (* Leaf-placement policy bindings (see su_schedule's comment). *)
  let load_policy =
    match Knobs.sched_loads () with
    | Some "anyorder" -> `Anyorder
    | Some "lookahead" -> `Lookahead
    | Some "nearstarve" -> `Nearstarve
    | _ -> `Strict
  in
  let leaf_pace =
    match Knobs.load_pace () with
    | Some s ->
      (try max 1 (int_of_string s) with
       | _ -> 4)
    | None -> 4
  in
  let since_leaf = ref 0 in
  let is_const n =
    match n.Ir.node with
    | NK_Const _ -> true
    | _ -> false
  in
  let is_leaf n = is_load n || (load_policy = `Lookahead && is_const n) in
  let is_hot n =
    List.exists
      (fun (u : Ir.t) ->
         (try Hashtbl.find unsched_count u.tag with
          | Not_found -> 0)
         = 1)
      (try Hashtbl.find users n.Ir.tag with
       | Not_found -> [])
  in
  let pick_next () : Ir.t option =
    if !ready = []
    then None
    else (
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
        let sa = Hashtbl.mem sink_set a.Ir.tag in
        let sb = Hashtbl.mem sink_set b.Ir.tag in
        if sa <> sb
        then compare sb sa (* true (sink) before false *)
        else (
          let cpa =
            try Hashtbl.find cp_dist a.Ir.tag with
            | Not_found -> 0
          in
          let cpb =
            try Hashtbl.find cp_dist b.Ir.tag with
            | Not_found -> 0
          in
          if cpa <> cpb
          then compare cpb cpa
          else (
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
              | `P5 -> !p01_fired - !p5_fired
              | `Other -> 0
            in
            let pba = port_balance_score a in
            let pbb = port_balance_score b in
            if pba <> pbb
            then compare pbb pba (* higher score first *)
            else (
              match tiebreak_sub with
              | `Affinity when Hashtbl.length sub_mask > 0 ->
                let ov t =
                  pc64s
                    (Int64.logand
                       !sub_last
                       (try Hashtbl.find sub_mask t with
                        | Not_found -> 0L))
                in
                let oa = ov a.tag
                and ob = ov b.tag in
                if oa <> ob then compare ob oa else compare a.tag b.tag
              | `Cone when Hashtbl.length sub_cone > 0 ->
                let ca =
                  try Hashtbl.find sub_cone a.tag with
                  | Not_found -> 0
                in
                let cb =
                  try Hashtbl.find sub_cone b.tag with
                  | Not_found -> 0
                in
                if ca <> cb then compare ca cb else compare a.tag b.tag
              | _ ->
                let sua =
                  try Hashtbl.find su_num a.tag with
                  | Not_found -> 0
                in
                let sub =
                  try Hashtbl.find su_num b.tag with
                  | Not_found -> 0
                in
                if sua <> sub then compare sua sub else compare a.tag b.tag)))
      in
      (* Goodman-Hsu pressure-mode comparator: minimize live-delta.
       *   delta(n) = births(n) - kills(n)
       *   kills(n) = preds in subset with remaining_users == 1 (n kills them)
       *   births(n) = 1 if n has remaining users in subset OR is a sink
       * Pick the most-negative delta (frees most relative to creates).
       * Tiebreak with cp_dist DESC (some latency awareness within pressure mode)
       * then tag ASC for stability. *)
      let pressure_delta n =
        let preds_in_sub =
          List.filter (fun p -> Hashtbl.mem in_subset p.Ir.tag) (preds n)
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
      let cmp_pressure a b =
        let da = pressure_delta a in
        let db = pressure_delta b in
        if da <> db
        then compare da db (* most-negative first *)
        else (
          let cpa =
            try Hashtbl.find cp_dist a.Ir.tag with
            | Not_found -> 0
          in
          let cpb =
            try Hashtbl.find cp_dist b.Ir.tag with
            | Not_found -> 0
          in
          if cpa <> cpb then compare cpb cpa else compare a.tag b.tag)
      in
      let cmp = if gh && live_count () > threshold then cmp_pressure else cmp_latency in
      (* === LEAF-PLACEMENT POLICY (env-gated; same modes as su_schedule,
       * see the comment there). strict = original byte-identical path;
       * anyorder = starvation firing over ALL ready loads by comparator;
       * lookahead = loads AND NK_Const deferrable, hot leaves paced in. *)
      match load_policy with
      | `Strict ->
        (match arith_ready with
         | _ :: _ -> Some (List.hd (List.sort cmp arith_ready))
         | [] ->
           let req_load = next_required_load () in
           let load_candidates =
             List.filter (fun n -> is_load n && Some n.Ir.tag = req_load) !ready
           in
           (match load_candidates with
            | [] -> None
            | _ -> Some (List.hd load_candidates)))
      | `Anyorder ->
        (match arith_ready with
         | _ :: _ -> Some (List.hd (List.sort cmp arith_ready))
         | [] ->
           (match List.filter is_load !ready with
            | [] -> None
            | ls -> Some (List.hd (List.sort cmp ls))))
      | `Lookahead ->
        let leaves, arith = List.partition is_leaf !ready in
        let hot_leaves = List.filter is_hot leaves in
        if !since_leaf >= leaf_pace && hot_leaves <> []
        then (
          since_leaf := 0;
          Some (List.hd (List.sort cmp hot_leaves)))
        else (
          match arith with
          | _ :: _ ->
            incr since_leaf;
            Some (List.hd (List.sort cmp arith))
          | [] ->
            since_leaf := 0;
            (match leaves with
             | [] -> None
             | ls -> Some (List.hd (List.sort cmp ls))))
      | `Nearstarve ->
        (* Fire the best HOT load when the arithmetic pool is nearly dry
         * (|arith_ready| <= VFFT_LOAD_PACE) — one to a few slots BEFORE
         * starvation would force it, instead of at starvation. Loads
         * only; constants keep their strict (arith-pool) behavior. *)
        let arith = List.filter (fun n -> not (is_load n)) !ready in
        let hot_loads = List.filter is_hot (List.filter is_load !ready) in
        if List.length arith <= leaf_pace && hot_loads <> []
        then Some (List.hd (List.sort cmp hot_loads))
        else (
          match arith with
          | _ :: _ -> Some (List.hd (List.sort cmp arith))
          | [] ->
            (match List.filter is_load !ready with
             | [] -> None
             | ls -> Some (List.hd (List.sort cmp ls)))))
  in
  let result : Ir.t list ref = ref [] in
  let rec loop () =
    match pick_next () with
    | None -> ()
    | Some n ->
      Hashtbl.add scheduled n.Ir.tag ();
      ready := List.filter (fun x -> x.Ir.tag <> n.tag) !ready;
      Hashtbl.remove in_ready n.tag;
      if is_load n then incr load_idx;
      (* Update port-class counters for port-balance tiebreaker. *)
      (match port_class n with
       | `P01 -> incr p01_fired
       | `P5 -> incr p5_fired
       | `Other -> ());
      result := n :: !result;
      (if tiebreak_sub <> `Classic
       then
         sub_last
         := try Hashtbl.find sub_mask n.tag with
            | Not_found -> !sub_last);
      (* Goodman-Hsu live-set update.
       * 1. For each pred P in subset, decrement P's remaining_users.
       *    If P hits 0 and isn't a sink, P dies (remove from live).
       * 2. n itself becomes live if it has remaining users OR is a sink. *)
      if gh
      then (
        let preds_in_sub =
          List.filter (fun p -> Hashtbl.mem in_subset p.Ir.tag) (preds n)
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
          try Hashtbl.find remaining_users n.tag with
          | Not_found -> 0
        in
        let is_sink = Hashtbl.mem sink_set n.tag in
        if n_users > 0 || is_sink then Hashtbl.replace live n.tag ());
      let user_list =
        try Hashtbl.find users n.tag with
        | Not_found -> []
      in
      List.iter
        (fun u ->
           let cur =
             try Hashtbl.find unsched_count u.Ir.tag with
             | Not_found -> 0
           in
           let new_count = cur - 1 in
           Hashtbl.replace unsched_count u.tag new_count;
           if
             new_count = 0
             && (not (Hashtbl.mem in_ready u.tag))
             && not (Hashtbl.mem scheduled u.tag)
           then (
             ready := u :: !ready;
             Hashtbl.add in_ready u.tag ()))
        user_list;
      loop ()
  in
  loop ();
  let su_order = List.rev !result in
  (* === SCHEDULE-SEARCH KNOBS for BLOCKED codelets (per-subset; default off) ===
   * su_schedule_subset is called once per PASS/group, so dump/inject are keyed by
   * a deterministic hash of the subset's sorted tags -> each pass gets its own
   * order file "<prefix>_<key>.txt". Same intent as the su_schedule knobs, keyed.
   * Illegal/incomplete injected orders fall back to su_order.
   * Wisdom-grade additions (see SCHEDULE WISDOM PLUMBING): dumps carry a
   * #dagsig computed over IN-SUBSET edges only (matching what the dump
   * describes); injection resolves through order_source, verifies the dagsig,
   * and logs every accept/refuse keyed by subset. subset_key only NAMES the
   * file; the dagsig is the staleness check. *)
  let subset_key =
    Hashtbl.hash (List.sort compare (List.map (fun (n : Ir.t) -> n.tag) subset))
  in
  let subset_sig () =
    dag_signature
      (List.map
         (fun (n : Ir.t) ->
            ( n.tag
            , List.filter_map
                (fun (p : Ir.t) ->
                   if Hashtbl.mem in_subset p.tag then Some p.tag else None)
                (preds n) ))
         su_order)
  in
  (match Knobs.Trace.sched_dump () with
   | None -> ()
   | Some prefix ->
     let oc = open_out (Printf.sprintf "%s_%d.txt" prefix subset_key) in
     Printf.fprintf oc "#dagsig %s\n" (subset_sig ());
     List.iter
       (fun (n : Ir.t) ->
          let ps =
            List.filter (fun (p : Ir.t) -> Hashtbl.mem in_subset p.tag) (preds n)
          in
          Printf.fprintf
            oc
            "%d:%s\n"
            n.tag
            (String.concat " " (List.map (fun (p : Ir.t) -> string_of_int p.tag) ps)))
       su_order;
     close_out oc);
  match resolve_order_source () with
  | None -> su_order
  | Some prefix ->
    let f = Printf.sprintf "%s_%d.txt" prefix subset_key in
    if not (Sys.file_exists f)
    then su_order
    else (
      let tags, file_sig = read_order_file f in
      match file_sig with
      | Some s when s <> subset_sig () ->
        log_injection
          (Printf.sprintf
             "subset %d: sched order STALE (dagsig mismatch), fell back to su: %s"
             subset_key
             f);
        su_order
      | _ ->
        let by_tag : (int, Ir.t) Hashtbl.t = Hashtbl.create 256 in
        List.iter (fun (n : Ir.t) -> Hashtbl.replace by_tag n.tag n) subset;
        let ordered = List.filter_map (fun t -> Hashtbl.find_opt by_tag t) tags in
        if List.length ordered = List.length su_order
        then (
          log_injection
            (Printf.sprintf
               "subset %d: sched order injected: %s (nodes=%d, %s)"
               subset_key
               f
               (List.length ordered)
               (match file_sig with
                | Some _ -> "dagsig verified"
                | None -> "no dagsig, unverified"));
          ordered)
        else (
          log_injection
            (Printf.sprintf
               "subset %d: sched order INCOMPLETE (%d of %d nodes), fell back to su: %s"
               subset_key
               (List.length ordered)
               (List.length su_order)
               f);
          su_order))
;;
