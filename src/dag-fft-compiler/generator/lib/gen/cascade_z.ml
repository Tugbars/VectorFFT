(* cascade_z.ml — M8.1: the Cascade_z feature module (§9 #35), renamed from
 * codelet_zsplit.ml — content unchanged (the doc's priced expectation: this
 * sub-step "cuts nothing and carries no risk signal"; zsplit was already the
 * best-behaved emitter).  Pipeline-hosted zil BLOCK-SPLIT split family.
 *
 * The port of codelet_zil.ml's raw-template split kinds onto the production
 * DAG pipeline (docs/roadmap/zil_pipeline_port.md). Where the legacy module
 * emits hand-written C strings, this one derives the SAME kernels through:
 *
 *   Dft.dft_expand_twiddled (math layer, elem-ref space)
 *     -> Pipeline.prepare_codelet (hash-cons + algsimp/FMA cascade)
 *     -> Schedule.su_schedule (SR list scheduler over the DAG)
 *     -> Emit_render.render_node_def (Isa-parameterized rendering)
 *
 * with zsplit-specific EDGES (block-split plane loads/stores) and TWIDDLE
 * RENDERING (Emit_state.current_tw_zsplit: [c×VW][s×VW] records in tw_re;
 * the z ABI's tw_im slot is dead).
 *
 * ABI: the frozen 11-arg z ABI of zsplit.h — function names, parameter
 * names, and (void) silencers match codelet_zil byte-for-byte so emitted
 * files are drop-in replacements in codelets/zil/avx2/.
 *
 * TIER GATE: the split family is radix 4/8 ONLY and monolithic BY DESIGN
 * (16 planes fit the ymm file; "r16 split = 32 live planes, spills" —
 * codelet_zil.ml). We therefore do NOT consult Dft.should_spill here:
 * its n>=5 clause would put R=8 on the spill recipe and pay stack traffic
 * the legacy kernels don't have (R1 op-census parity would fail).
 *
 * Kinds (zil_pipeline_port.md §8, P1–P5): ms/msb (in-place split mid),
 * msg/msgb (group-looped wrapper), s0s/s0sb (z-boundary leaf), sterm/stermb
 * (terminator), sterm2 (2-quad unroll-and-jam terminator = TWO shifted DAG
 * instances concatenated il2-style and braided by the SU scheduler, plus a
 * baseline-shaped 4-column tail). Bwd kinds run with ~table_conj because
 * the runtime tables (twspb/twqb) are ALREADY conjugated — the double-conj
 * trap, zil_pipeline_port.md §6.1.
 * ------------------------------------------------------------------
 * MODULE CARD (codelet_zsplit.ml — grep "MODULE CARD" for the full set)
 * ROLE: DAG-pipeline emitter for the zil block-split kinds.
 * PIPELINE: Dft -> Pipeline.prepare_codelet -> Schedule.su_schedule ->
 * zsplit edges + Emit_render.render_node_def.
 * PUBLIC SURFACE: emit_codelet (gen_main --zp-* flags).
 * DEPS: Dft, Algsimp, Pipeline, Schedule, Emit_c (render + provenance +
 * compute_inline_set + current_tw_zsplit via the Emit_state chain), Isa,
 * Uarch.
 * GOTCHA 1: sets Emit_state.current_tw_zsplit for the duration of emission
 * (Fun.protect-reset); no other family may leave it non-None.
 * GOTCHA 2: sterm2 prepares TWO DAGs sequentially (main 2-instance, then
 * the 1-instance tail); each prepare calls Ir.reset, so the main
 * body must be FULLY rendered to the buffer before the tail's prepare.
 * ------------------------------------------------------------------ *)

(* ─── kind table ──────────────────────────────────────────────────── *)

(* Edge shapes for the column loop's memory boundary:
     E_planes — block-split planes: re at 2*(l*S+k), im +VW. Plain vector
                loads/stores, shuffle-free (the interior contract).
     E_z      — natural interleaved z at the same leg addressing: two z
                vectors per leg, DEINT on load / REINT on store (the API
                boundary; shuffles paid once per cascade).
     E_blocks — terminator col-block edge: per column, R consecutive
                complexes as R/VW [re×VW][im×VW] blocks at 2·R·(k+c);
                TR4 register transposes swap column-lane ↔ leg-index
                (load: blocks → leg-major lanes; store: the inverse).
   Each stride-using edge carries the C stride NAME ("Ls" | "OLs") —
   the leaf/mids run on Ls, the terminator's z comb on OLs. *)
type zs_edge =
  | E_planes of string
  | E_z of string
  | E_blocks
  | E_sect_tap of string
    (* ZTURN-S sectioned-record taps (r0=4): leg q -> section SEC[q&3],
       SEC = bitrev2 = {0,2,1,3}; sections at s*4*STRIDE doubles; 2
       consecutive 64-B records per section per column group (base
       4*(size_t)k, second record +8, im +VW). Plain loadu/storeu,
       shuffle-free. stf load side / stfb store side. *)
  | E_sect_tr4 of string
(* ZTURN-S backward-ingest load network: 4 section records at
       leg_addr(l = SEC[m], STRIDE) + two tr4_str lane transposes
       (the un-turn). LOAD-ONLY. *)

(* ─── B2 store/load placement policies (--zp-sched) ────────────────
   Memory operations become first-class SCHEDULED nodes (ZNode below);
   in B2 their POSITIONS are PINNED to reproduce the existing outputs
   byte-for-byte — the cost-driven placement SEARCH is B3:
     ZS_legacy   — [preamble loads] ++ [arith in EXACTLY su_schedule's
                   order] ++ [stores in edge order]: the committed
                   placement, as a real node sequence.
     ZS_afterdef — each singleton store unit immediately after its
                   sink's def (the B1 --zp-sink shape, on the sequence
                   path; reproduces radix8_z_stf_r4sk_bwd_avx2's body).
   ZS_off = the pre-B2 code path, byte-untouched (the default). *)
type zs_sched =
  | ZS_off
  | ZS_legacy
  | ZS_afterdef

type zs_kind =
  { base : string (* "ms" | "msg" | "s0s" | "sterm" | "sterm2" — C-name stem *)
  ; bwd : bool
  ; group_loop : bool
    (* msg: emit the ms column loop as a static always_inline _zsg body
       and export a thin wrapper that walks the Gs groups in-kernel
       (bp += 2·R·Ls doubles, twg += (R-1)·2·VW). One call per stage —
       kills the per-group call overhead + trip-count mispredicts
       (z_cascade_plan §4.9991/§4.9992). *)
  ; uj2 : bool
    (* sterm2: 2-quad unroll-and-jam. The math layer concatenates TWO
       shifted instances (Input/Output +R slots, Twiddle +1 slot — quad
       B's packed w¹ record is exactly the next 2·VW doubles of the
       stream, so the render mode composes for free) and the SU scheduler
       braids them by readiness — the pipeline's answer to the hand
       template's 5 phases (z_cascade_plan §4.9993). Main loop k += 2·VW
       plus a baseline-shaped VW-column tail. Their kernels are
       bit-identical by construction; which one the cascade runs is the
       MEASURED per-cell t2q pick (§4.9994). *)
  ; twiddled : bool (* false: n1 math + (void)tw_re (s0s leaf) *)
  ; policy : Dft.twiddle_policy
    (* TP_Flat: splat-pair records per leg (mids). TP_PowW1: ONE packed
       per-column w¹ record, higher powers derived in-DAG by the
       squaring tree (terminator). *)
  ; tw_off : string
    (* C base-offset expression for the twiddle record stream: "" for
       table-start records (mids), "2*(size_t)k" for the terminator's
       column-indexed packed-w¹ stream. *)
  ; in_edge : zs_edge
  ; out_edge : zs_edge
  ; sink_stores : bool
    (* B1 store-sinking: interleave the store edge into the SU-scheduled
       body — each sink's store is emitted the moment its value node is
       defined, instead of the whole edge trailing the body (su_schedule
       is edge-blind; a trailing edge keeps all 2·R sink values live to
       loop end). Semantics precedent: Emit_state.current_store_on_compute
       (PASS-2 blocked path) — output value nodes are sinks, their only
       use is the store, so storing at def is the natural last use.
       OPT-IN per kind via gen_main --zp-sink (fname gains an "sk" tag);
       NEVER a kind-table default — the default regeneration of every
       kind stays byte-identical to the committed files. Only store edges
       whose per-slot stores are SINGLETON-ready (one direct storeu per
       sink tag, no cross-slot shuffle/transpose network) can sink;
       today that is E_sect_tap alone (stfb). *)
  ; nat_in : bool
    (* NATURAL-ORDER variant (stfn/stfbn -- docs/research/natterm_spec.md):
       the IN-side edge and (fwd only) the twiddle stream are addressed
       through `kn`, a per-iteration column index read from a plan-built
       rho-inverse (fwd) / rho (bwd) block table carried in via the UNUSED
       tw_im arg (cast to const size_t ptr). Stores keep `k` ascending --
       contiguous natural output. Load-side permutation is the measured-free
       side (P0c: 0.96-1.12x vs +29-50% for store-side). *)
  ; nat_out : bool
    (* STORE-side permutation twin of nat_in (dtso): the OUT-side edge and
       the twiddle stream are addressed through kn; LOADS keep k ascending
       (contiguous user reads). Only legal when the OUT edge writes the
       PLANE (E_sect_tap) -- scattered 64B stores into the L2-resident
       plane, the cheap side; scattered stores to USER memory are the
       P0c-condemned side and no kind may do that. *)
  ; dif : bool
    (* twiddle PLACEMENT direction for dft_expand_twiddled. false = DIT
       (pre-twiddle at Fwd — every legacy kind). true = DIF (POST-twiddle at
       Fwd) — required by the DIT-forward boundary kinds: conj(stfb) is
       "DFT block THEN w^1", and placement travels with (direction, sign)
       in dft.ml:271-277, so sign=Fwd alone lands on PRE and computes a
       DIFFERENT (wrong) kernel. Found by the conj-identity gate: dts/dtsn
       grossly wrong (maxabs ~ output magnitude) while the twiddle-free dtt
       was EXACT — the placement, not the block, was the bug. *)
  ; sched : zs_sched
    (* B2 --zp-sched: emit through the combined memory+arith node
       sequence (see zs_sched above). NEVER a kind-table default —
       set only by emit_codelet from the flag, so default regeneration
       stays on the pre-B2 path (not even entered). *)
  }

let kind_of_string (s : string) : zs_kind =
  let mid =
    { base = "ms"
    ; bwd = false
    ; group_loop = false
    ; uj2 = false
    ; twiddled = true
    ; policy = Dft.TP_Flat
    ; tw_off = ""
    ; in_edge = E_planes "Ls"
    ; out_edge = E_planes "Ls"
    ; sink_stores = false
    ; sched = ZS_off
    ; nat_in = false
    ; nat_out = false
    ; dif = false
    }
  in
  match s with
  | "ms" -> mid
  | "msb" -> { mid with bwd = true }
  | "msg" -> { mid with base = "msg"; group_loop = true }
  | "msgb" -> { mid with base = "msg"; group_loop = true; bwd = true }
  | "msd" ->
    (* DIT-FORWARD mid = conj(msgb) (dit_cascade_spec.md): msg's group-loop
       dataflow with DIF placement — DFT block THEN post-cmul(twz as
       loaded). NOT the msg fwd kernel (that is DIT pre-twiddle): the same
       placement trap the boundary kinds hit, caught by the pipeline gate
       (kernels conj-EXACT, composition grossly wrong). *)
    { mid with base = "msd"; group_loop = true; dif = true }
  | "s0s" -> { mid with base = "s0s"; twiddled = false; in_edge = E_z "Ls" }
  | "s0sb" -> { mid with base = "s0s"; twiddled = false; bwd = true; out_edge = E_z "Ls" }
  | "sterm" ->
    { mid with
      base = "sterm"
    ; policy = Dft.TP_PowW1
    ; tw_off = "2*(size_t)k"
    ; in_edge = E_blocks
    ; out_edge = E_z "OLs"
    }
  | "stermb" ->
    { mid with
      base = "sterm"
    ; bwd = true
    ; policy = Dft.TP_PowW1
    ; tw_off = "2*(size_t)k"
    ; in_edge = E_z "OLs"
    ; out_edge = E_blocks
    }
  | "sterm2" ->
    (* fwd only: the bwd 2-quad was REFUTED (+29..36%% kernel, §4.9993). *)
    { mid with
      base = "sterm2"
    ; uj2 = true
    ; policy = Dft.TP_PowW1
    ; tw_off = "2*(size_t)k"
    ; in_edge = E_blocks
    ; out_edge = E_z "OLs"
    }
  (* ── ZTURN-S sectioned kinds (docs/roadmap/cascade_load_path_restructure.md
        Amendment ZTURN-S; prototype = src/core/oop/zturn_proto.h). Bases are
        "s0t"/"stf" — NEVER "sterm": fname derives from base, and reusing the
        incumbent stem would overwrite its generated files (both routes must
        stay linkable side by side). *)
  | "s0t" ->
    (* fwd = closed-form template (emit_s0t_body); edges here feed ONLY
       uses/plain_voids: loads are natural z at leg addressing (Ls),
       stores are sectioned records (Ls). *)
    { mid with
      base = "s0t"
    ; twiddled = false
    ; in_edge = E_z "Ls"
    ; out_edge = E_sect_tr4 "Ls"
    }
  | "s0tb" ->
    { mid with
      base = "s0t"
    ; bwd = true
    ; twiddled = false
    ; in_edge = E_sect_tr4 "Ls"
    ; out_edge = E_z "Ls"
    }
  | "stf" ->
    (* radix 8 (last==8 chains) AND radix 4 (last==4 chains — the ZTURN-S
       radix-4 terminator, r4term_sim derivation E6-E11: same edges/policy,
       radix-parametric sect_addr, OLs = count = N/4, tzq N/16 groups). *)
    { mid with
      base = "stf"
    ; policy = Dft.TP_PowW1
    ; tw_off = "2*(size_t)k"
    ; in_edge = E_sect_tap "OLs"
    ; out_edge = E_z "OLs"
    }
  | "stfb" ->
    { mid with
      base = "stf"
    ; bwd = true
    ; policy = Dft.TP_PowW1
    ; tw_off = "2*(size_t)k"
    ; in_edge = E_z "OLs"
    ; out_edge = E_sect_tap "OLs"
    }
  | "stf2" ->
    (* fwd ONLY (the sterm2 precedent — zsplit.h's bwd always runs the
       single-quad sterm_bwd; the bwd 2-quad was REFUTED, +29..36%%
       kernel, §4.9993 — so no stfb2). stf's edges/policy/tw_off + uj2:
       the 4th create-time race arm (t2q) for the ZTURN-S route,
       roadmap §3.2(g/h). Instance B = the NEXT section-record group
       (+vw columns = +8 plane doubles per section tap) and the NEXT
       packed w^1 record (Twiddle slot +1 = +2*vw doubles) — composing
       exactly as sterm2's streamed table. *)
    { mid with
      base = "stf2"
    ; uj2 = true
    ; policy = Dft.TP_PowW1
    ; tw_off = "2*(size_t)k"
    ; in_edge = E_sect_tap "OLs"
    ; out_edge = E_z "OLs"
    }
  | "stfn" ->
    (* NATURAL-ORDER terminator (fwd): stf edges with the IN side (section
       taps) AND the packed-w1 stream addressed at kn = 4*tbl[k>>2]
       (tbl = rho-inverse, MSF digit reversal over the middle radices --
       P0b), stores contiguous ascending = natural interleaved output.
       Radix 8 AND radix 4, like stf. No uj2 twin (t2q pinned 0 in natural
       phase 1). *)
    { mid with
      base = "stfn"
    ; policy = Dft.TP_PowW1
    ; tw_off = "2*(size_t)kn"
    ; in_edge = E_sect_tap "OLs"
    ; out_edge = E_z "OLs"
    ; nat_in = true
    }
  | "stfbn" ->
    (* NATURAL-ORDER bwd terminator: consumes a NATURAL spectrum -- the z
       (user) LOAD edge reads at kn (tbl = rho: the value scrambled-bwd
       expects at block k sits at natural block rho(k)); the conj-w1 stream
       and the section-tap plane STORES keep k (they belong to the scrambled
       column being reconstructed). *)
    { mid with
      base = "stfn"
    ; bwd = true
    ; policy = Dft.TP_PowW1
    ; tw_off = "2*(size_t)k"
    ; in_edge = E_z "OLs"
    ; out_edge = E_sect_tap "OLs"
    ; nat_in = true
    }
  (* ── DIT-FORWARD family (docs/research/dit_cascade_spec.md). Algebra:
        F = conj . B . conj, and conjugation flips only constant signs, so a
        DIT-forward kernel = an existing BWD kind's DATAFLOW (same edges, same
        addresses) emitted with sign=Fwd and table_conj=false (bwd=false does
        both), consuming the plan's FWD tables (twz/tzq — elementwise conj of
        twzb/tzqb). The mids need NO new kind: conj(msgb) = msg exactly, so
        the DIT driver walks the EXISTING msg fwd kernels in the bwd stage
        order. Only the two boundaries are new, and both are recombinations —
        no emitter-mechanics change. Bit-identity gate is exact BY
        CONSTRUCTION: dts(x, tzq) == conj(stfb(conj x, tzqb)), memcmp. *)
  | "dts" ->
    (* DIT ingest (user -> plane): stfb's dataflow at fwd sign. Reads the
       user z buffer in DIGIT-REVERSED column order (the DIT-native input
       arrangement), DFT + POST w^1 (TP_PowW1 stream at k), section-record
       stores. Radix 8 AND 4, like stf. *)
    { mid with
      base = "dts"
    ; policy = Dft.TP_PowW1
    ; tw_off = "2*(size_t)k"
    ; in_edge = E_z "OLs"
    ; out_edge = E_sect_tap "OLs"
    ; dif = true
    }
  | "dtsn" ->
    (* DIT ingest, NATURAL-input twin: dts with the z LOAD edge addressed at
       kn via the tw_im-carried rho table (the stfbn mechanism, conj'd) --
       natural x in, no separate reorder. tw/w^1 stream and plane stores
       keep k. *)
    { mid with
      base = "dtsn"
    ; policy = Dft.TP_PowW1
    ; tw_off = "2*(size_t)k"
    ; in_edge = E_z "OLs"
    ; out_edge = E_sect_tap "OLs"
    ; nat_in = true
    ; dif = true
    }
  | "dtso" ->
    (* DIT ingest, STORE-side-permutation twin of dtsn: user z LOADS stay
       contiguous at k (the whole point -- reclaim the rho-scattered
       user-memory reads the stage probe blamed for the race loss); the
       section-record STORES and the w^1 stream go to kn via the tw_im
       table = rho^{-1} (ntf). Plane contents IDENTICAL to dtsn's --
       iteration k does what dtsn's iteration rho^{-1}(k) does -- so the
       gate is plane memcmp vs dtsn. *)
    { mid with
      base = "dtso"
    ; policy = Dft.TP_PowW1
    ; tw_off = "2*(size_t)kn"
    ; in_edge = E_z "OLs"
    ; out_edge = E_sect_tap "OLs"
    ; nat_out = true
    ; dif = true
    }
  | "dtt" ->
    (* DIT finisher (plane -> user): s0tb's dataflow at fwd sign.
       Twiddle-free leaf, section-record loads + 4x4 lane transpose, REINT
       stores contiguous ascending = NATURAL interleaved output with zero
       indirection on the store side. Radix 4 only (r0=4 geometry). *)
    { mid with
      base = "dtt"
    ; twiddled = false
    ; in_edge = E_sect_tr4 "Ls"
    ; out_edge = E_z "Ls"
    }
  | other ->
    failwith
      (Printf.sprintf
         "codelet_zsplit: unknown kind %s (supported: ms msb msg msgb s0s s0sb sterm \
          stermb sterm2 s0t s0tb stf stfb stf2 stfn stfbn dts dtsn dtt msd dtso)"
         other)
;;

(* ─── B2: memory operations as first-class scheduled nodes ─────────
 *
 * The zsplit COMBINED graph: the arith DAG (Algsimp, scheduled by
 * su_schedule) plus address-carrying memory pseudo-nodes for the edge
 * fragments the pre-B2 code rendered out-of-band (loads in a preamble,
 * stores in a trailing edge / the B1 flush). Instantiation precedent:
 * codelet_cil's own Node + Schedule.Make — extending Ir.node_kind was
 * priced at ~150 match arms and REJECTED there; same verdict here.
 *
 * HONESTY LEDGER (what the B2 graph does and does not model):
 *   - ZStore preds = the wrapped sink value nodes: REAL edges (a store
 *     fires only after its operands exist). Granularity = one store
 *     UNIT (zs_store_unit below): the smallest fragment that can fire
 *     once its sink tags are defined.
 *   - ZLoad preds = [] AND no arith node points at a ZLoad: the value
 *     edge from the address-carrying preamble unit to the NK_Load
 *     value-copies is NOT modeled yet — wiring it (so a search can
 *     move loads) is B3 work. In B2 every policy pins loads to the
 *     preamble, where that edge is trivially satisfied.
 *   - Tags: arith keeps its Algsimp tags; memory nodes get tags above
 *     the arith maximum (loads first, then stores), preserving the
 *     tag-ascending-is-topological invariant the scheduler's DPs rely
 *     on if the combined graph is ever scheduled (ZStores sit topmost).
 *)

(* One pre-rendered store-edge fragment. The B1 sink_pending table,
   generalized per edge shape; ALL store consumers render from it —
   the trailing edge, B1's store-at-def flush, and the B2 sequence
   walk — so there is exactly one source of store bytes. *)
type zs_store_unit =
  { su_tags : int list (* sink value tags this unit retires (fire when all defined) *)
  ; su_orefs :
      Expr.elem_ref list (* the Output refs behind su_tags (assigns-side identity) *)
  ; su_text : string (* the fragment's C, byte-identical to the pre-B2 rendering *)
  }

type zpayload =
  | ZArith of Ir.t (* a scheduled arith node, by reference *)
  | ZLoad of { unit_idx : int } (* one load-edge preamble unit *)
  | ZStore of
      { unit_idx : int
      ; sinks : Ir.t list (* the wrapped sink nodes = preds *)
      }

module ZNode : Schedule.SCHED_NODE with type payload = zpayload = struct
  type payload = zpayload

  type t =
    { tag : int
    ; node : payload
    }

  let preds (n : t) : t list =
    match n.node with
    | ZArith e ->
      List.map (fun (p : Ir.t) -> { tag = p.Ir.tag; node = ZArith p }) (Ir.preds e)
    | ZLoad _ -> [] (* address roots; the value edge to NK_Load is B3 (see ledger) *)
    | ZStore { sinks; _ } ->
      List.map (fun (s : Ir.t) -> { tag = s.Ir.tag; node = ZArith s }) sinks
  ;;

  let latency (u : Uarch.t) (n : t) : int =
    match n.node with
    | ZArith e -> Schedule.node_latency u e
    | ZLoad _ -> u.Uarch.load_l1_latency
    | ZStore _ -> u.Uarch.store_latency (* retire cost, mostly hidden *)
  ;;

  (* NK_Load value-copies KEEP their load class (delegation), so a
     combined-graph SR run starve-gates them exactly as the Ir run does;
     ZLoad units are loads too. *)
  let is_load (n : t) =
    match n.node with
    | ZArith e -> Schedule.Ir_node.is_load e
    | ZLoad _ -> true
    | ZStore _ -> false
  ;;

  let is_store (n : t) =
    match n.node with
    | ZStore _ -> true
    | ZArith _ | ZLoad _ -> false
  ;;

  let is_const (n : t) =
    match n.node with
    | ZArith e -> Schedule.Ir_node.is_const e
    | ZLoad _ | ZStore _ -> false
  ;;

  let kind_char (n : t) =
    match n.node with
    | ZArith e -> Schedule.Ir_node.kind_char e
    | ZLoad _ -> 'L'
    | ZStore _ -> 'S'
  ;;
end

(* The SHARED SR scheduler over the combined graph. B2 never ships its
   order (policies are pinned); the VFFT_ZP_SCHED_SU knob below runs it
   as a B3 seed instrument. *)
module ZSched = Schedule.Make (ZNode)

(* ─── emission ────────────────────────────────────────────────────── *)

let emit_codelet
      ~store_on_compute
      ~(kind : string)
      ~(radix : int)
      ~(r0 : int option)
      ~(sink_stores : bool)
      ~(sched : string option)
      ~(isa : Isa.t)
      ~(uarch : Uarch.t)
  : string
  =
  (* M6.1: per-emission scratch — this family's own instance *)
  let sc = Emit_render.Scratch.create () in
  (* M6.2: this family's config view — tw is FORWARD-passed, not a back-edge;
     the Tw_zsplit payload is set at the point the record offset is known. *)
  let cfg = ref { Emit_render.Cfg.default with Emit_render.Cfg.store_on_compute } in
  let k = kind_of_string kind in
  (* ── --zp-sink admission gate (the r0_dep-gate precedent: fail loudly,
        never silently emit the unsunk shape under the sunk name). Only a
        SINGLETON-ready store edge can sink — E_sect_tap's direct record
        stores (stfb). TR4/REINT edges (E_blocks, E_z, E_planes) need a
        readiness-SET interleave (a store fires only when its whole
        shuffle network's inputs are defined) — NEW emitter work. ── *)
  let k =
    if not sink_stores
    then k
    else (
      match k.out_edge with
      | E_sect_tap _ -> { k with sink_stores = true }
      | E_planes _ | E_z _ | E_blocks | E_sect_tr4 _ ->
        failwith
          (Printf.sprintf
             "codelet_zsplit: --zp-sink: kind %s's store edge is not sink-capable (only \
              E_sect_tap's direct record stores — stfb — sink today; TR4/REINT store \
              edges need a readiness-set interleave, which is NEW emitter work, not a \
              flag)"
             kind))
  in
  (* ── --zp-sched admission gate (B2): the combined memory+arith node
        sequence. Loud on: --zp-sink together with --zp-sched (two flags,
        one output shape — an A/B script would label one arm twice), a
        bogus policy string (a typo must not silently ship a pinned shape
        under a searched-sounding name), and afterdef on a store edge
        that is not sink-capable (exactly the --zp-sink constraint). ── *)
  let k =
    match sched with
    | None -> k
    | Some _ when sink_stores ->
      failwith
        "codelet_zsplit: --zp-sink and --zp-sched request the same stores through two \
         mechanisms (B1 flush vs B2 sequence); pass exactly one (--zp-sched afterdef IS \
         the B1 store-at-def shape, sequence-hosted)"
    | Some "legacy" ->
      (* Universal by design: legacy is the reproduction policy every kind
         must pass through (G-1) before B3 searches anything. On s0t fwd
         (closed-form template, no DAG) it is the identity. *)
      { k with sched = ZS_legacy }
    | Some "afterdef" ->
      (match k.out_edge with
       | E_sect_tap _ ->
         (* sink_stores=true reuses B1's fname "sk" tag and body-header
            text, so the sequence path reproduces the committed sk kernel
            under the same symbol (A/B side by side with the incumbent). *)
         { k with sched = ZS_afterdef; sink_stores = true }
       | E_planes _ | E_z _ | E_blocks | E_sect_tr4 _ ->
         failwith
           (Printf.sprintf
              "codelet_zsplit: --zp-sched afterdef: kind %s's store edge is not \
               sink-capable (only E_sect_tap's direct record stores — stfb — sink today, \
               the --zp-sink constraint; TR4/REINT store edges need a readiness-set \
               interleave, which is B3 emitter work, not a policy string)"
              kind))
    | Some other ->
      failwith
        (Printf.sprintf
           "codelet_zsplit: --zp-sched %s: unknown placement policy (policies: legacy = \
            preamble loads ++ SU-order arith ++ trailing stores, byte-reproduces the \
            committed files; afterdef = each store at its sink's def, the --zp-sink \
            shape). B3 adds searched placements; a typo must not silently ship a pinned \
            shape."
           other)
  in
  (* ── r0 is a PLAN INPUT (chain[0]), never chosen here. The ZTURN-S kinds
        bake the 4-section geometry into every address, so their fname MUST
        carry it (a future r0=8 twin sharing the symbol would silently ship
        a wrong FFT). Legacy kinds must not carry it (provenance stability). *)
  let r0_dep =
    k.base = "s0t"
    || k.base = "stf"
    || k.base = "stf2"
    || k.base = "stfn"
    || k.base = "dts"
    || k.base = "dtsn"
    || k.base = "dtt"
    || k.base = "dtso"
  in
  (match r0, r0_dep with
   | None, true ->
     failwith
       "codelet_zsplit: s0t/stf bake r0 into their section addressing; pass --zp-r0 \
        (plan INPUT, chain[0]; no default)"
   | Some r, true when r <> 4 ->
     failwith
       (Printf.sprintf
          "codelet_zsplit: only the r0=4 ZTURN-S geometry is emitted (got %d); an r0=%d \
           twin is NEW emitter work, not a parameter change"
          r
          r)
   | Some _, false ->
     failwith
       "codelet_zsplit: --zp-r0 is only for the ZTURN-S kinds; legacy kinds must not \
        carry it (provenance stability)"
   | _ -> ());
  if radix <> 4 && radix <> 8
  then failwith "codelet_zsplit: split family is radix 4/8 only (see TIER GATE)";
  if (k.base = "sterm" || k.base = "sterm2") && radix <> 8
  then failwith "codelet_zsplit: sterm/sterm2 are radix-8 only (zsplit.h chain contract)";
  (* stf/stfb emit at radix 8 AND radix 4 (the ZTURN-S radix-4 terminator,
     r4term_sim derivation E6-E15: ONE 64-B record per section per group,
     count = OLs = N/4, truncated squaring tree w^2 = w^2, w^3 = w^2*w).
     stf2 has NO r4 twin: the 2-quad instance B's "+1 section-record group"
     offset presumes 2 records/group (radix 8); at radix 4 zturn.h forces
     t2q = 0 and the planner races CHAINS instead. *)
  if k.base = "stf2" && radix <> 8
  then
    failwith
      "codelet_zsplit: stf2 is radix-8 only — there is NO stf2@r4 twin (the radix-4 \
       terminator taps ONE record/section, so instance B's +1-record-group column offset \
       has no analog; zturn.h forces t2q=0 for last==4 chains and the planner races \
       chains instead)";
  if (k.base = "s0t" || k.base = "dtt") && radix <> 4
  then
    failwith "codelet_zsplit: s0t/s0tb/dtt are radix-4 only (the r0=4 4-section geometry)";
  let vw = isa.Isa.vec_width in
  if vw <> 4
  then
    (* The generator side is width-parameterized, but the RUNTIME block
       geometry ([re×VW][im×VW]) is baked into zsplit.h's plan builder at
       VW=4. Lift this gate together with the zsplit.h vw parameterization
       (zil_pipeline_port.md §5). *)
    failwith
      "codelet_zsplit: runtime block geometry is VW=4 until zsplit.h is parameterized";
  (* ⚠ uj2 × E_sect_tap column-offset coincidence (roadmap §3.2(h)): the
     2-quad instance B sits at colo = vw columns, which lands on "+1
     section-record group" (+2*vw plane doubles per tap) ONLY because
     r0 == vw. An r0=8 twin needs an 8-column unroll, not this flag —
     ASSERTED, not assumed. *)
  if
    k.uj2
    &&
    match k.in_edge with
    | E_sect_tap _ -> true
    | E_planes _ | E_z _ | E_blocks | E_sect_tr4 _ -> false
  then (
    match r0 with
    | Some r when r = vw -> ()
    | _ ->
      failwith
        (Printf.sprintf
           "codelet_zsplit: stf2's uj2 instance-B column offset (+vw=%d) equals one \
            section-record group only when r0 == vw; r0 <> vw needs a new unroll (NEW \
            emitter work, not a parameter change)"
           vw));
  let dir_s = if k.bwd then "bwd" else "fwd" in
  let r0_tag =
    match r0 with
    | Some r -> Printf.sprintf "_r%d" r
    | None -> ""
  in
  (* the r0_tag mechanism, reused: the sunk variant is a DIFFERENT kernel
     shape and must stay linkable next to the incumbent (A/B side by
     side), so its fname carries "sk" (radix8_z_stf_r4sk_bwd_avx2). *)
  let sink_tag = if k.sink_stores then "sk" else "" in
  let fname =
    Printf.sprintf
      "radix%d_z_%s%s%s_%s_%s"
      radix
      k.base
      r0_tag
      sink_tag
      dir_s
      isa.Isa.name
  in
  let sign : [ `Fwd | `Bwd ] = if k.bwd then `Bwd else `Fwd in
  let force_fma_lift =
    try Sys.getenv "VFFT_FORCE_FMA_LIFT" = "1" with
    | Not_found -> false
  in
  let disable_fma_lift =
    try Sys.getenv "VFFT_DISABLE_FMA_LIFT" = "1" with
    | Not_found -> false
  in
  (* ─── DAG preparation (math layer + shared cascade + SU schedule) ──
     Called once per emitted body. sterm2 calls it twice — 2-instance
     main, then 1-instance tail — see MODULE CARD GOTCHA 2.
     ms fwd: DIT pre-twiddle t1 (legs 1..R-1 cmul'd on load) — exactly
     dft_expand_twiddled's (DIT, Fwd). msb: (DIT, Bwd) = IDFT butterfly +
     POST-twiddle, with ~table_conj because twspb carries the conjugation.
     s0s/s0sb: twiddle-free n1 (the leaf pays only the z boundary).
     sterm/sterm2: TP_PowW1 squaring tree from the packed w¹ record. *)
  let prepare ~(two_inst : bool)
    : (Expr.elem_ref * Ir.t) list
      * (Expr.elem_ref option * Ir.t) list
      * (int, unit) Hashtbl.t
    =
    let base_assigns =
      if k.twiddled
      then
        Dft.dft_expand_twiddled
          ~policy:k.policy
          ~direction:(if k.dif then Dft.DIF else Dft.DIT)
          ~sign
          ~table_conj:k.bwd
          radix
      else Dft.dft_expand ~sign radix
    in
    let raw_assigns =
      if not two_inst
      then base_assigns
      else (
        (* il2-style second instance: Input/Output slots +radix, Twiddle
           slots +1 (TP_PowW1 consults slot 0 only, so instance B reads
           slot 1 = the NEXT packed record at +2·VW doubles — matching
           quad B's w¹ in the streamed table). Mirror of dft.ml's
           dft_expand_twiddled_il2, with the PowW1 twiddle shift. *)
        let shift_ref = function
          | Expr.Input (i, p) -> Expr.Input (i + radix, p)
          | Expr.Output (i, p) -> Expr.Output (i + radix, p)
          | Expr.Twiddle (t, p) -> Expr.Twiddle (t + 1, p)
        in
        let rec shift = function
          | Expr.Const c -> Expr.Const c
          | Expr.Load r -> Expr.Load (shift_ref r)
          | Expr.Neg e -> Expr.Neg (shift e)
          | Expr.Add (a, b) -> Expr.Add (shift a, shift b)
          | Expr.Sub (a, b) -> Expr.Sub (shift a, shift b)
          | Expr.Mul (a, b) -> Expr.Mul (shift a, shift b)
        in
        base_assigns @ List.map (fun (lhs, e) -> shift_ref lhs, shift e) base_assigns)
    in
    Ir.reset ();
    let pipe : Pipeline.prepared =
      Pipeline.prepare_codelet
        ~recipe:Pipeline.default_recipe
        ~raw_assigns
        ~spill_markers_raw:[]
        ~spill_ct:None
        ~reassoc:
          (match Sys.getenv_opt "VFFT_FORCE_REASSOC" with
           | Some "0" -> false
           | Some "1" -> true
           | _ -> Dft_select.needs_reassoc radix)
        ~aggressive:false
        ~algorithm:(Dft_select.pick_algorithm radix)
        ~force_fma_lift
        ~disable_fma_lift
        ~build_spill_info:false
        ~fuse:0
    in
    let assigns = pipe.Pipeline.assigns in
    let scheduled = Schedule.su_schedule uarch assigns in
    let inline_set = Emit_render.compute_inline_set ~sc assigns in
    assigns, scheduled, inline_set
  in
  (* ─── emit ────────────────────────────────────────────────────────── *)
  let buf = Buffer.create 8192 in
  Buffer.add_string
    buf
    (Printf.sprintf
       "/* Auto-generated by vfft_v2 — BLOCK-SPLIT interior family, PIPELINE-HOSTED\n\
       \ * (codelet_zsplit.ml; docs/roadmap/zil_pipeline_port.md). Scratch = 64-B\n\
       \ * [re x%d][im x%d] blocks (z addressing +%d for im; one stream per leg row).\n\
       \ * %s\n\
       \ * CONTRACT: count %% %d == 0 (%s).\n\
       \ * %s%s */\n"
       vw
       vw
       vw
       (match k.base, k.bwd with
        | "sterm2", _ ->
          "sterm2: 2-quad unroll-and-jam terminator twin (SU-braided 2-instance DAG + \
           baseline-shaped tail; bit-identical pair with sterm, per-cell t2q pick)."
        | "sterm", false ->
          "sterm (SPLIT-INPUT terminator: TR4 loads, packed w^1 squaring tree, REINT \
           drev-comb stores), fwd."
        | "sterm", true ->
          "sterm bwd twin (drev comb DEINT in, IDFT + POST conj-w^1, TR4 block stores)."
        | "s0s", false -> "s0s (z-in -> split-out leaf, twiddle-free, DEINT loads), fwd."
        | "s0s", true ->
          "s0s bwd twin (split-in -> natural-z-out IDFT leaf, REINT stores)."
        | "msg", false ->
          "msg (GROUP-LOOPED split mid: one call/stage, in-kernel bp/twg bumps), fwd."
        | "msg", true ->
          "msg bwd twin (group loop over IDFT+POST-tw body; table twspb pre-conjugated)."
        | "s0t", false ->
          "s0t (ZTURN-S fused-turn ingest: natural z leg loads, twiddle-free radix-4, \
           ONE 64-B record per position at section bitrev2(p mod 4), 4 rate-matched \
           cursors), fwd."
        | "s0t", true ->
          "s0tb (ZTURN-S bwd ingest: section-record loads + 4x4 lane transpose un-turn, \
           IDFT-4, REINT natural-z stores)."
        | "stf", false ->
          if radix = 8
          then
            "stf (ZTURN-S terminator: 4 section taps, 2 consecutive records = 128 B \
             contiguous per tap, NO load shuffles, packed w^1 squaring tree, REINT \
             drev-comb stores), fwd."
          else
            "stf@r4 (ZTURN-S RADIX-4 terminator: 4 section taps, ONE 64-B record per \
             tap, NO load shuffles, packed w^1 truncated squaring tree, REINT drev-comb \
             stores; OLs = count = N/4), fwd."
        | "stf", true ->
          if radix = 8
          then
            "stfb (ZTURN-S bwd terminator: drev comb DEINT in, IDFT + POST conj-w^1, \
             DIRECT record stores at section taps — no TR4)."
          else
            "stfb@r4 (ZTURN-S RADIX-4 bwd terminator = the bwd FIRST stage: drev comb \
             DEINT in, IDFT-4 + POST conj-w^1, DIRECT one-record stores at section taps \
             — no TR4; OLs = count = N/4)."
        | "stf2", _ ->
          "stf2: 2-quad unroll-and-jam ZTURN-S terminator twin (SU-braided 2-instance \
           DAG + baseline-shaped tail; section-tap loads, instance B at +1 record group; \
           bit-identical pair with stf, per-cell t2q pick)."
        | "stfn", false ->
          "stfn (NATURAL-ORDER ZTURN-S terminator: section-tap loads + packed w^1 stream \
           at kn = 4*rhoinv[k/4] via the tw_im-carried table, REINT stores contiguous \
           ascending = natural interleaved out), fwd."
        | "stfn", true ->
          "stfbn (NATURAL-ORDER ZTURN-S bwd terminator: NATURAL z in, read at kn = \
           4*rho[k/4] via the tw_im-carried table, IDFT + POST conj-w^1 at k, \
           section-record stores at k), bwd."
        | "msd", false ->
          "msd (DIT-FORWARD mid = conj(msgb): group loop over DFT + POST-twiddle body, \
           fwd table twz as loaded), fwd."
        | "dts", false ->
          "dts (DIT-FORWARD ZTURN-S ingest = conj(stfb): user z in DIGIT-REVERSED column \
           order, DFT + POST w^1 (packed stream at k, FWD tzq table), section-record \
           stores), fwd."
        | "dtsn", false ->
          "dtsn (DIT-FORWARD ZTURN-S ingest, NATURAL-input twin = conj(stfbn): user z \
           read at kn = 4*rho[k/4] via the tw_im-carried table, DFT + POST w^1 at k, \
           section-record stores at k), fwd."
        | "dtso", false ->
          "dtso (DIT-FORWARD ZTURN-S ingest, STORE-side permutation twin: user z read \
           CONTIGUOUS at k, DFT + POST w^1 at kn, section-record stores at kn = \
           4*rhoinv[k/4] via the tw_im-carried table -- scatter confined to the hot \
           plane), fwd."
        | "dtt", false ->
          "dtt (DIT-FORWARD ZTURN-S finisher = conj(s0tb): twiddle-free leaf, \
           section-record loads + 4x4 lane transpose, REINT stores contiguous ascending \
           = NATURAL interleaved out), fwd."
        | _, true ->
          "ms bwd twin (IDFT + POST-twiddle; table twspb pre-conjugated -> table_conj)."
        | _, false ->
          "ms (split mid, IN-PLACE zin==zout, SHUFFLE-FREE, splat-pair tw), fwd.")
       vw
       (if k.uj2
        then Printf.sprintf "%d columns per main trip, %d-column tail" (2 * vw) vw
        else Printf.sprintf "%d columns per iteration" vw)
       (if not k.twiddled
        then "tw_re/tw_im unused (twiddle-free leaf)."
        else if k.policy = Dft.TP_PowW1
        then
          Printf.sprintf
            "tw_re = packed per-column w^1 at tw_re + 2k: [c(k..k+%d)][s(k..k+%d)]; \
             powers w^2..w^%d derived in-register (squaring tree). tw_im unused."
            (vw - 1)
            (vw - 1)
            (radix - 1)
        else
          Printf.sprintf
            "tw_re = %s: legs 1..R-1, %d doubles/leg [c×%d][s×%d]. tw_im unused."
            (if k.group_loop
             then "Gs per-group splat-pair sets, in-kernel cursor (twg bump/group)"
             else "ONE per-group splat-pair set, no cursor (group-constant)")
            (2 * vw)
            vw
            vw)
       (if k.bwd then " Roundtrip = N*x (no 1/N in-kernel)." else ""));
  Buffer.add_string
    buf
    (Emit_render.provenance_block
       ~family:"zsplit-pipeline"
       [ Printf.sprintf
           "kind=%s radix=%d dir=%s isa=%s%s%s"
           k.base
           radix
           dir_s
           isa.Isa.name
           (match r0 with
            | Some r -> Printf.sprintf " r0=%d (ZTURN-S sectioned geometry)" r
            | None -> "")
           (* B2: ZS_legacy stamps NOTHING here — its job is byte-
              reproduction of the committed files (the argv line above
              already records the flag); afterdef reaches sink-stores
              through the sequence path, so the token names the flag
              that actually ran. *)
           (match k.sched with
            | ZS_afterdef -> " sink-stores (store-at-def interleave, --zp-sched afterdef)"
            | ZS_off | ZS_legacy ->
              if k.sink_stores
              then " sink-stores (store-at-def interleave, --zp-sink)"
              else "")
       ; (if k.twiddled
          then
            Printf.sprintf
              "math: Dft.dft_expand_twiddled %s %s%s%s"
              (match k.policy with
               | Dft.TP_PowW1 -> "TP_PowW1"
               | Dft.TP_Log3 -> "TP_Log3"
               | Dft.TP_Flat -> "TP_Flat")
              (if k.dif then "DIF" else "DIT")
              (if k.bwd then " sign=Bwd table_conj=true" else " sign=Fwd")
              (if k.uj2 then " (2-instance concat, il2-style)" else "")
          else "math: Dft.dft_expand (n1)" ^ if k.bwd then " sign=Bwd" else " sign=Fwd")
       ; "prepare: Pipeline.prepare_codelet (monolithic, fuse=0)"
       ; "schedule: Schedule.su_schedule (SR list scheduler)"
       ]);
  Buffer.add_string buf "#include <immintrin.h>\n#include <stddef.h>\n\n";
  (* s0t fwd's turn lattice needs the (im,re)-swap sign mask for x*(-i);
     file-scope const so gcc hoists ONE load out of the loop (prototype's
     _vzt_mim, zturn_proto.h). *)
  if k.base = "s0t" && not k.bwd
  then Buffer.add_string buf (Isa.im_mask_decl isa "_zs0t_mim" ^ "\n\n");
  (* TR4 rendering helper (E_blocks): 4 unpacks + 4 permute2f128 turning
     four column vectors into four leg/index vectors (or back). srcs/dsts
     are C variable names; dsts are declared const. Returns the fragment
     (B2: edges render through chunk sinks; pre-B2 call sites passed the
     same two sprintf blocks straight to the buffer — the concatenation
     is byte-identical). *)
  let tr4_str ~(qid : string) (srcs : string array) (dsts : string array) : string =
    let unlo = Isa.intr isa "unpacklo_pd"
    and unhi = Isa.intr isa "unpackhi_pd"
    and p2f = Isa.intr isa "permute2f128_pd" in
    Printf.sprintf
      "        %s\n        %s\n        %s\n        %s\n"
      (Isa.const_decl
         isa
         (Printf.sprintf "_u0_%s" qid)
         (Printf.sprintf "%s(%s, %s)" unlo srcs.(0) srcs.(1)))
      (Isa.const_decl
         isa
         (Printf.sprintf "_u1_%s" qid)
         (Printf.sprintf "%s(%s, %s)" unhi srcs.(0) srcs.(1)))
      (Isa.const_decl
         isa
         (Printf.sprintf "_u2_%s" qid)
         (Printf.sprintf "%s(%s, %s)" unlo srcs.(2) srcs.(3)))
      (Isa.const_decl
         isa
         (Printf.sprintf "_u3_%s" qid)
         (Printf.sprintf "%s(%s, %s)" unhi srcs.(2) srcs.(3)))
    ^ Printf.sprintf
        "        %s\n        %s\n        %s\n        %s\n"
        (Isa.const_decl
           isa
           dsts.(0)
           (Printf.sprintf "%s(_u0_%s, _u2_%s, 0x20)" p2f qid qid))
        (Isa.const_decl
           isa
           dsts.(1)
           (Printf.sprintf "%s(_u1_%s, _u3_%s, 0x20)" p2f qid qid))
        (Isa.const_decl
           isa
           dsts.(2)
           (Printf.sprintf "%s(_u0_%s, _u2_%s, 0x31)" p2f qid qid))
        (Isa.const_decl
           isa
           dsts.(3)
           (Printf.sprintf "%s(_u1_%s, _u3_%s, 0x31)" p2f qid qid))
  in
  (* column-c block address: base 2·R·(k+c), halves at h·2·VW, im +VW *)
  let blk_addr (buf_name : string) (c : int) (off : int) : string =
    if c = 0
    then Printf.sprintf "%s[%d*(size_t)k + %d]" buf_name (2 * radix) off
    else Printf.sprintf "%s[%d*((size_t)k + %d) + %d]" buf_name (2 * radix) c off
  in
  (* leg-addressed edge address: 2*(leg*STRIDE + k + colo), im/hi +VW.
     colo is the instance's column offset (0 for instance A). *)
  let leg_addr
        ?(iv = "k")
        (buf_name : string)
        (leg : int)
        (stride : string)
        (colo : int)
        (plus : int)
    : string
    =
    let base =
      match leg, colo with
      | 0, 0 -> Printf.sprintf "2*(size_t)%s" iv
      | 0, o -> Printf.sprintf "2*((size_t)%s + %d)" iv o
      | l, 0 -> Printf.sprintf "2*((size_t)%d*%s + %s)" l stride iv
      | l, o -> Printf.sprintf "2*((size_t)%d*%s + %s + %d)" l stride iv o
    in
    if plus = 0
    then Printf.sprintf "%s[%s]" buf_name base
    else Printf.sprintf "%s[%s + %d]" buf_name base plus
  in
  (* ZTURN-S sectioned-record address, r0=4 BAKED IN (hence fname _r4):
     leg q -> section SEC[q land 3] (SEC = bitrev2 = 0,2,1,3), im +VW.
     RADIX-PARAMETRIC over the terminator radix (records per section per
     column group = radix/4, address scale = radix/2 with STRIDE = OLs =
     N/radix — both collapse to the section double-offset sec*N/2):
       radix 8: 2 consecutive records (base 4k, second +8, 128 B/tap);
       radix 4: ONE record (base 2k, 64 B/tap) — derivation (E6):
                leg q of group k2 taps record SEC[q]*S + k2, S = N/16.
     colo = the uj2 instance's column offset (0 for instance A — renders
     byte-identically to the pre-stf2 form; vw for instance B = the next
     section-record group, legal because r0 == vw, asserted above; uj2
     is radix-8-only, so the r4 form never sees colo <> 0). *)
  let sect_addr
        ?(iv = "k")
        (buf_name : string)
        (q : int)
        (stride : string)
        (colo : int)
        (plus : int)
    : string
    =
    let sec = [| 0; 2; 1; 3 |].(q land 3) in
    let sc = radix / 2 in
    let off = (8 * (q asr 2)) + plus in
    match sec, colo with
    | 0, 0 -> Printf.sprintf "%s[%d*(size_t)%s + %d]" buf_name sc iv off
    | 0, o -> Printf.sprintf "%s[%d*((size_t)%s + %d) + %d]" buf_name sc iv o off
    | s, 0 ->
      Printf.sprintf "%s[%d*((size_t)%d*%s + %s) + %d]" buf_name sc s stride iv off
    | s, o ->
      Printf.sprintf "%s[%d*((size_t)%d*%s + %s + %d) + %d]" buf_name sc s stride iv o off
  in
  (* natural-order in-side index: `kn` (declared per iteration, nat_in only). *)
  let ivin = if k.nat_in then "kn" else "k" in
  (* nat_out (dtso): the OUT-side edge walks kn instead. *)
  let ivout = if k.nat_out then "kn" else "k" in
  (* ── column loop: edges + SU-scheduled body + store edge for ONE
        prepared DAG. ninst = 1 normally, 2 for the sterm2 main loop
        (slots radix..2·radix-1 are instance B = columns k+VW..k+2·VW-1).
        open_line is the C for-statement (the uj2 shell shares one
        function-scope k across the main loop and the tail). ── *)
  let emit_col_loop
        ~(open_line : string)
        ~(ninst : int)
        ((assigns, scheduled, inline_set) :
          (Expr.elem_ref * Ir.t) list
          * (Expr.elem_ref option * Ir.t) list
          * (int, unit) Hashtbl.t)
    : unit
    =
    let nslots = ninst * radix in
    Buffer.add_string buf open_line;
    if k.nat_in || k.nat_out
    then
      Buffer.add_string
        buf
        "        /* natural-order: in-side block index via the rho table (tw_im \
         repurposed; natterm_spec.md) */\n\
        \        const size_t kn = 4*((const size_t *)tw_im)[(size_t)k >> 2];\n";
    (* ── B2 chunk sinks: every load-edge fragment is rendered through a
          sink function. Flag-off (ZS_off): the sinks ARE
          Buffer.add_string buf, called in the exact order of the pre-B2
          code — the byte path is unchanged (gate G-3). Flag-on: the
          fragments are captured as ZLoad units and the combined-sequence
          walk below is the only writer of the loop body. ── *)
    let sched_on = k.sched <> ZS_off in
    let load_hdr_s = ref "" in
    let load_units_rev : string list ref = ref [] in
    let load_hdr (s : string) : unit =
      if sched_on then load_hdr_s := s else Buffer.add_string buf s
    in
    let load_chunk (s : string) : unit =
      if sched_on then load_units_rev := s :: !load_units_rev else Buffer.add_string buf s
    in
    (match k.in_edge with
     | E_planes s ->
       (* ── ZBlockSplit load edge: lane_{re,im}_l from the split planes.
             Leg l's re half at zin + 2*(l*S + k), im half +VW. ── *)
       load_hdr "        /* ZBlockSplit load edge */\n";
       for sl = 0 to nslots - 1 do
         let leg = sl mod radix
         and colo = sl / radix * vw in
         load_chunk
           (Printf.sprintf
              "        %s\n        %s\n"
              (Isa.const_decl
                 isa
                 (Printf.sprintf "lane_re_%d" sl)
                 (Isa.loadu_pd isa (leg_addr "zin" leg s colo 0)))
              (Isa.const_decl
                 isa
                 (Printf.sprintf "lane_im_%d" sl)
                 (Isa.loadu_pd isa (leg_addr "zin" leg s colo vw))))
       done
     | E_z s ->
       (* ── Z load edge (DEINT): two z vectors per leg, deinterleaved into
             the lane planes — unpacklo/hi + permute4x64 0xD8, the shuffles
             the cascade pays once at its API boundary. ── *)
       load_hdr "        /* Z load edge (DEINT) */\n";
       let unlo = Isa.intr isa "unpacklo_pd"
       and unhi = Isa.intr isa "unpackhi_pd"
       and p44 = Isa.intr isa "permute4x64_pd" in
       for sl = 0 to nslots - 1 do
         let leg = sl mod radix
         and colo = sl / radix * vw in
         load_chunk
           (Printf.sprintf
              "        %s\n        %s\n        %s\n        %s\n"
              (Isa.const_decl
                 isa
                 (Printf.sprintf "_zl_%d" sl)
                 (Isa.loadu_pd isa (leg_addr ~iv:ivin "zin" leg s colo 0)))
              (Isa.const_decl
                 isa
                 (Printf.sprintf "_zh_%d" sl)
                 (Isa.loadu_pd isa (leg_addr ~iv:ivin "zin" leg s colo vw)))
              (Isa.const_decl
                 isa
                 (Printf.sprintf "lane_re_%d" sl)
                 (Printf.sprintf "%s(%s(_zl_%d, _zh_%d), 0xD8)" p44 unlo sl sl))
              (Isa.const_decl
                 isa
                 (Printf.sprintf "lane_im_%d" sl)
                 (Printf.sprintf "%s(%s(_zl_%d, _zh_%d), 0xD8)" p44 unhi sl sl)))
       done
     | E_blocks ->
       (* ── Block load edge (TR4): per column, load the R/VW block halves
             and TR4 each quad into leg-major lanes (lane = column). The
             terminator's load-side transpose; ninst=2 covers 2·VW cols. ── *)
       load_hdr "        /* Block load edge (TR4) */\n";
       let halves = radix / vw in
       for c = 0 to (ninst * vw) - 1 do
         for h = 0 to halves - 1 do
           load_chunk
             (Printf.sprintf
                "        %s\n        %s\n"
                (Isa.const_decl
                   isa
                   (Printf.sprintf "_br%d_%d" h c)
                   (Isa.loadu_pd isa (blk_addr "zin" c (h * 2 * vw))))
                (Isa.const_decl
                   isa
                   (Printf.sprintf "_bi%d_%d" h c)
                   (Isa.loadu_pd isa (blk_addr "zin" c ((h * 2 * vw) + vw)))))
         done
       done;
       for inst = 0 to ninst - 1 do
         for h = 0 to halves - 1 do
           load_chunk
             (tr4_str
                ~qid:(Printf.sprintf "lr%d_%d" h inst)
                (Array.init 4 (fun j -> Printf.sprintf "_br%d_%d" h ((inst * vw) + j)))
                (Array.init 4 (fun j ->
                   Printf.sprintf "lane_re_%d" ((inst * radix) + (h * vw) + j))));
           load_chunk
             (tr4_str
                ~qid:(Printf.sprintf "li%d_%d" h inst)
                (Array.init 4 (fun j -> Printf.sprintf "_bi%d_%d" h ((inst * vw) + j)))
                (Array.init 4 (fun j ->
                   Printf.sprintf "lane_im_%d" ((inst * radix) + (h * vw) + j))))
         done
       done
     | E_sect_tap s ->
       (* ── ZTURN-S section-tap load edge: leg q taps section SEC[q&3];
             radix/4 consecutive 64-B records per section per column group
             (r8: 2 records = 128 B contiguous; r4: ONE record = 64 B, the
             derivation's E6 map), lanes pass straight through — NO load
             shuffles (the stf side of the turn's payoff). ── *)
       load_hdr
         (if radix = 8
          then
            "        /* ZTURN-S section-tap load edge (2 records/section, 128 B \
             contiguous, no shuffles) */\n"
          else
            "        /* ZTURN-S section-tap load edge (1 record/section, 64 B, no \
             shuffles) */\n");
       (* ninst = 2 (stf2): slots radix..2*radix-1 are instance B = the
          NEXT section-record group at colo = vw columns (r0 == vw
          asserted at the admission gate). ninst = 1 renders slot = q,
          colo = 0 — byte-identical to the pre-stf2 form. *)
       for sl = 0 to nslots - 1 do
         let q = sl mod radix
         and colo = sl / radix * vw in
         load_chunk
           (Printf.sprintf
              "        %s\n        %s\n"
              (Isa.const_decl
                 isa
                 (Printf.sprintf "lane_re_%d" sl)
                 (Isa.loadu_pd isa (sect_addr ~iv:ivin "zin" q s colo 0)))
              (Isa.const_decl
                 isa
                 (Printf.sprintf "lane_im_%d" sl)
                 (Isa.loadu_pd isa (sect_addr ~iv:ivin "zin" q s colo vw))))
       done
     | E_sect_tr4 s ->
       (* ── ZTURN-S backward-ingest load edge: granule's 4 section records
             (section order SEC = its own inverse, so natural position
             order) + 4x4 lane transpose = the un-turn. ── *)
       load_hdr
         "        /* ZTURN-S section-record load edge + 4x4 lane transpose (un-turn) */\n";
       if ninst <> 1 || radix <> vw
       then failwith "codelet_zsplit: E_sect_tr4 assumes radix = VW = 4, ninst = 1";
       let sec = [| 0; 2; 1; 3 |] in
       for m = 0 to 3 do
         load_chunk
           (Printf.sprintf
              "        %s\n        %s\n"
              (Isa.const_decl
                 isa
                 (Printf.sprintf "_sr_%d" m)
                 (Isa.loadu_pd isa (leg_addr "zin" sec.(m) s 0 0)))
              (Isa.const_decl
                 isa
                 (Printf.sprintf "_si_%d" m)
                 (Isa.loadu_pd isa (leg_addr "zin" sec.(m) s 0 vw))))
       done;
       load_chunk
         (tr4_str
            ~qid:"lr0_0"
            (Array.init 4 (Printf.sprintf "_sr_%d"))
            (Array.init 4 (Printf.sprintf "lane_re_%d")));
       load_chunk
         (tr4_str
            ~qid:"li0_0"
            (Array.init 4 (Printf.sprintf "_si_%d"))
            (Array.init 4 (Printf.sprintf "lane_im_%d"))));
    (* per-slot output tag arrays (all edge shapes consume pairs) — built
       BEFORE the body walk so sink_stores can interleave stores at defs *)
    let re_tag = Array.make nslots (-1)
    and im_tag = Array.make nslots (-1) in
    List.iter
      (fun (lhs, (e : Ir.t)) ->
         match lhs with
         | Expr.Output (l, true) -> re_tag.(l) <- e.Ir.tag
         | Expr.Output (l, false) -> im_tag.(l) <- e.Ir.tag
         | _ -> failwith "codelet_zsplit: assign LHS is not Output")
      assigns;
    (* ── store-edge unit table: ONE pre-rendered table for every store
          consumer — the trailing edge, B1's store-at-def flush, and the
          B2 sequence walk all render from it (the B1 sink_pending table,
          generalized per edge shape; concatenating a table in order
          reproduces the pre-B2 trailing edge byte-for-byte). Unit
          granularity = the smallest fragment that can fire once its sink
          tags are defined: E_sect_tap per (slot, re/im) single storeu
          (SINGLETON — the only granularity afterdef threads); E_planes /
          E_z per slot; E_blocks per TR4 quad (the whole shuffle network
          + its 2·VW stores). An ARRAY, not a hashtable: if CSE ever
          aliases two slots to one tag, each slot keeps its own unit and
          both fire at that tag's def. ── *)
    let store_hdr, (store_units : zs_store_unit array) =
      match k.out_edge with
      | E_planes s ->
        ( "        /* ZBlockSplit store edge */\n"
        , Array.init nslots (fun sl ->
            let leg = sl mod radix
            and colo = sl / radix * vw in
            { su_tags = [ re_tag.(sl); im_tag.(sl) ]
            ; su_orefs = [ Expr.Output (sl, true); Expr.Output (sl, false) ]
            ; su_text =
                Printf.sprintf
                  "        %s;\n        %s;\n"
                  (Isa.storeu_pd
                     isa
                     (leg_addr "zout" leg s colo 0)
                     (Printf.sprintf "t%d" re_tag.(sl)))
                  (Isa.storeu_pd
                     isa
                     (leg_addr "zout" leg s colo vw)
                     (Printf.sprintf "t%d" im_tag.(sl)))
            }) )
      | E_z s ->
        (* ── Z store edge (REINT): permute4x64 0xD8 each plane, then
              unpacklo/hi re-interleaves back to natural z. For the
              terminator this addressing (leg-major on OLs = N/R) IS the
              digit-reversed comb — the scramble itself is plan-side. ── *)
        let unlo = Isa.intr isa "unpacklo_pd"
        and unhi = Isa.intr isa "unpackhi_pd"
        and p44 = Isa.intr isa "permute4x64_pd" in
        ( "        /* Z store edge (REINT) */\n"
        , Array.init nslots (fun sl ->
            let leg = sl mod radix
            and colo = sl / radix * vw in
            { su_tags = [ re_tag.(sl); im_tag.(sl) ]
            ; su_orefs = [ Expr.Output (sl, true); Expr.Output (sl, false) ]
            ; su_text =
                Printf.sprintf
                  "        %s\n        %s\n        %s;\n        %s;\n"
                  (Isa.const_decl
                     isa
                     (Printf.sprintf "_pr_%d" sl)
                     (Printf.sprintf "%s(t%d, 0xD8)" p44 re_tag.(sl)))
                  (Isa.const_decl
                     isa
                     (Printf.sprintf "_qi_%d" sl)
                     (Printf.sprintf "%s(t%d, 0xD8)" p44 im_tag.(sl)))
                  (Isa.storeu_pd
                     isa
                     (leg_addr "zout" leg s colo 0)
                     (Printf.sprintf "%s(_pr_%d, _qi_%d)" unlo sl sl))
                  (Isa.storeu_pd
                     isa
                     (leg_addr "zout" leg s colo vw)
                     (Printf.sprintf "%s(_pr_%d, _qi_%d)" unhi sl sl))
            }) )
      | E_blocks ->
        (* ── Block store edge (TR4 back): leg-major result vectors
              transpose to column vectors, stored as each column's R/VW
              block halves (the stermb output side). Unit = one (inst,h)
              quad: its two TR4 networks + stores fire together once all
              8 sink lanes are defined. ── *)
        let halves = radix / vw in
        ( "        /* Block store edge (TR4) */\n"
        , Array.init (ninst * halves) (fun u ->
            let inst = u / halves
            and h = u mod halves in
            let slot j = (inst * radix) + (h * vw) + j in
            let sr =
              tr4_str
                ~qid:(Printf.sprintf "sr%d_%d" h inst)
                (Array.init 4 (fun j -> Printf.sprintf "t%d" re_tag.(slot j)))
                (Array.init 4 (fun j -> Printf.sprintf "_cr%d_%d" h ((inst * vw) + j)))
            and si =
              tr4_str
                ~qid:(Printf.sprintf "si%d_%d" h inst)
                (Array.init 4 (fun j -> Printf.sprintf "t%d" im_tag.(slot j)))
                (Array.init 4 (fun j -> Printf.sprintf "_ci%d_%d" h ((inst * vw) + j)))
            in
            let stores =
              String.concat
                ""
                (List.init vw (fun j ->
                   let c = (inst * vw) + j in
                   Printf.sprintf
                     "        %s;\n        %s;\n"
                     (Isa.storeu_pd
                        isa
                        (blk_addr "zout" c (h * 2 * vw))
                        (Printf.sprintf "_cr%d_%d" h c))
                     (Isa.storeu_pd
                        isa
                        (blk_addr "zout" c ((h * 2 * vw) + vw))
                        (Printf.sprintf "_ci%d_%d" h c))))
            in
            { su_tags =
                List.init 4 (fun j -> re_tag.(slot j))
                @ List.init 4 (fun j -> im_tag.(slot j))
            ; su_orefs =
                List.init 4 (fun j -> Expr.Output (slot j, true))
                @ List.init 4 (fun j -> Expr.Output (slot j, false))
            ; su_text = sr ^ si ^ stores
            }) )
      | E_sect_tap s ->
        (* ── ZTURN-S direct record store edge (stfb): leg q's results go
              straight to record SEC[q&3]*..., 2k'+(q>>2) — the TR4 store
              edge is GONE (exact address-mirror of stf's loads). Unit =
              ONE storeu per (slot, re/im), q ascending, re then im:
              SINGLETON-ready, exactly what the --zp-sink/afterdef
              admission gate promises. ── *)
        if ninst <> 1
        then
          failwith
            "codelet_zsplit: E_sect_tap store edge has no uj2 form (the bwd 2-quad — \
             stfb2 — is REFUTED, +29..36% kernel, §4.9993; only the LOAD side unrolls, \
             stf2)"
        else
          ( "        /* ZTURN-S direct record store edge (no TR4) */\n"
          , Array.init (2 * nslots) (fun i ->
              let q = i / 2 in
              let tag, plus, isre =
                if i land 1 = 0 then re_tag.(q), 0, true else im_tag.(q), vw, false
              in
              { su_tags = [ tag ]
              ; su_orefs = [ Expr.Output (q, isre) ]
              ; su_text =
                  Printf.sprintf
                    "        %s;\n"
                    (Isa.storeu_pd
                       isa
                       (sect_addr ~iv:ivout "zout" q s 0 plus)
                       (Printf.sprintf "t%d" tag))
              }) )
      | E_sect_tr4 _ ->
        failwith
          "codelet_zsplit: E_sect_tr4 is load-only (s0t's record stores are \
           template-emitted)"
    in
    let sink_done = Array.make (Array.length store_units) false in
    let flush_sunk_stores (tag : int) : unit =
      Array.iteri
        (fun i (u : zs_store_unit) ->
           if (not sink_done.(i)) && u.su_tags = [ tag ]
           then (
             sink_done.(i) <- true;
             Buffer.add_string buf u.su_text))
        store_units
    in
    (* ── SU-scheduled body. Defs in schedule order (first occurrence per
          tag); single-use tags render inline at their consumer. Twiddle
          loads render via the zsplit record mode. sink_stores: each def
          whose tag is a pending sink immediately trails its store line
          (store-at-def = the value's natural last use, the
          Emit_state.current_store_on_compute thesis). ── *)
    let body_hdr =
      if k.sink_stores
      then "        /* SU-scheduled body (pipeline) + sect-tap stores sunk at defs */\n"
      else "        /* SU-scheduled body (pipeline) */\n"
    in
    if not sched_on
    then (
      (* ══ pre-B2 path (ZS_off): body walk (+ B1 flush) then trailing
             store edge / defensive residual — byte-for-byte the committed
             behavior (gate G-3). ══ *)
      Buffer.add_string buf body_hdr;
      Fun.protect
        ~finally:(fun () ->
          cfg := { !cfg with Emit_render.Cfg.tw = Emit_render.Cfg.Tw_default })
        (fun () ->
           cfg := { !cfg with Emit_render.Cfg.tw = Emit_render.Cfg.Tw_zsplit k.tw_off };
           let seen : (int, unit) Hashtbl.t = Hashtbl.create 256 in
           List.iter
             (fun ((_ : Expr.elem_ref option), (e : Ir.t)) ->
                if
                  (not (Hashtbl.mem seen e.Ir.tag))
                  && not (Hashtbl.mem inline_set e.Ir.tag)
                then (
                  Hashtbl.replace seen e.Ir.tag ();
                  (* render_node_def embeds its own 8-space indent *)
                  Buffer.add_string
                    buf
                    (Emit_render.render_node_def
                       ~sc
                       ~cfg:!cfg
                       ~isa
                       ~in_place:false
                       ~t1s:false
                       ~strided:true
                       ~inline_set:(Some inline_set)
                       e);
                  Buffer.add_char buf '\n';
                  if k.sink_stores then flush_sunk_stores e.Ir.tag))
             scheduled);
      if k.sink_stores
      then
        (* Stores were interleaved at defs; nothing should be pending here:
            sink tags are never inlined (compute_inline_set treats outputs
            as sinks — the current_store_on_compute comment's invariant), so
            every sink tag materialized as a named def in the walk above.
            Defensive flush, loudly marked — a line below this comment means
            the slot->tag tables disagree with the schedule. *)
        Array.iteri
          (fun i (u : zs_store_unit) ->
             if not sink_done.(i)
             then (
               sink_done.(i) <- true;
               Buffer.add_string
                 buf
                 "        /* UNSUNK RESIDUAL (slot->tag table / schedule mismatch) */\n";
               Buffer.add_string buf u.su_text))
          store_units
      else (
        (* trailing store edge = the unit table in order (the pre-B2
            per-edge rendering, pre-rendered above) *)
        Buffer.add_string buf store_hdr;
        Array.iter
          (fun (u : zs_store_unit) -> Buffer.add_string buf u.su_text)
          store_units))
    else (
      (* ══ B2 path (--zp-sched): the combined memory+arith sequence.
             The nodes are REAL (ZNode.t, honest preds/tags); their
             POSITIONS are the pinned policy — B3 replaces the pinning
             with measured search, nothing else changes shape. ══ *)
      let load_units = Array.of_list (List.rev !load_units_rev) in
      let arith_order =
        List.filter_map
          (fun (o, (e : Ir.t)) ->
             match o with
             | None -> Some e
             | Some _ ->
               (* su_schedule's appended assign stores: the sequence's
                   ZStore units REPLACE this legacy representation *)
               None)
          scheduled
      in
      let base = 1 + List.fold_left (fun m (e : Ir.t) -> max m e.Ir.tag) 0 arith_order in
      let by_tag : (int, Ir.t) Hashtbl.t = Hashtbl.create 256 in
      List.iter (fun (e : Ir.t) -> Hashtbl.replace by_tag e.Ir.tag e) arith_order;
      let nload = Array.length load_units in
      let zarith (e : Ir.t) : ZNode.t = { ZNode.tag = e.Ir.tag; node = ZArith e } in
      let zload (i : int) : ZNode.t =
        { ZNode.tag = base + i; node = ZLoad { unit_idx = i } }
      in
      let zstore (j : int) : ZNode.t =
        let sinks =
          List.filter_map (fun t -> Hashtbl.find_opt by_tag t) store_units.(j).su_tags
        in
        { ZNode.tag = base + nload + j; node = ZStore { unit_idx = j; sinks } }
      in
      let loads = List.init nload zload in
      let residual : (int, unit) Hashtbl.t = Hashtbl.create 4 in
      let combined : ZNode.t list =
        match k.sched with
        | ZS_legacy ->
          (* [preamble loads] ++ [arith in EXACTLY su_schedule's order]
              ++ [stores in edge order] — the committed placement as a
              real node sequence (gate G-1). *)
          loads
          @ List.map zarith arith_order
          @ List.init (Array.length store_units) zstore
        | ZS_afterdef ->
          (* B1 semantics (gate G-2): each SINGLETON unit threads
              immediately after its sink's RENDERED def — the inline
              check mirrors the B1 flush firing only after rendered
              defs (sink tags are never inlined, so in practice every
              unit threads). Unthreaded units trail as loud residuals. *)
          let used = Array.make (Array.length store_units) false in
          let seq = ref [] in
          List.iter
            (fun (e : Ir.t) ->
               seq := zarith e :: !seq;
               if not (Hashtbl.mem inline_set e.Ir.tag)
               then
                 Array.iteri
                   (fun j (u : zs_store_unit) ->
                      if (not used.(j)) && u.su_tags = [ e.Ir.tag ]
                      then (
                        used.(j) <- true;
                        seq := zstore j :: !seq))
                   store_units)
            arith_order;
          Array.iteri
            (fun j _ ->
               if not used.(j)
               then (
                 Hashtbl.replace residual j ();
                 seq := zstore j :: !seq))
            store_units;
          loads @ List.rev !seq
        | ZS_off -> assert false (* sched_on *)
      in
      (* ── EXPERIMENTAL (B3 seed; default OFF, no gate depends on it):
             run the SHARED SR scheduler over the combined graph with the
             ZStore units as first-class sinks. Under SR's own rules a
             ready ZStore is an empty-user sink, so RETIRE fires it at
             readiness — after-def placement EMERGES instead of being
             pinned. ZLoad roots have no users yet (see ZNode's honesty
             ledger), so the run covers arith+stores; load-edge wiring is
             B3. Dump is append-mode (sterm2 emits two loops per file);
             sinks appear once as intermediates and once as su_schedule's
             trailing assigns echo. Shares the process env with the
             Ir-side VFFT_SCHED_DUMP hooks — don't set both. ── *)
      (match Sys.getenv_opt "VFFT_ZP_SCHED_SU" with
       | None -> ()
       | Some file ->
         let store_assigns =
           List.init (Array.length store_units) (fun j ->
             ( (match store_units.(j).su_orefs with
                | r :: _ -> r
                | [] -> failwith "codelet_zsplit: store unit with no output ref")
             , zstore j ))
         in
         let order = ZSched.su_schedule uarch store_assigns in
         let oc = open_out_gen [ Open_append; Open_creat ] 0o644 file in
         Printf.fprintf
           oc
           "# %s combined-graph SR order (%d rows)\n"
           fname
           (List.length order);
         List.iter
           (fun ((_ : Expr.elem_ref option), (n : ZNode.t)) ->
              Printf.fprintf oc "%d %c\n" n.ZNode.tag (ZNode.kind_char n))
           order;
         close_out oc);
      (* ── sequence-driven emission: ONE walk renders loads, arith and
             stores in sequence order, dispatching through the SCHED_NODE
             accessors. Headers fire at each class's first node — under
             the pinned policies that reproduces the pre-B2 header
             positions exactly. ── *)
      let seen : (int, unit) Hashtbl.t = Hashtbl.create 256 in
      let load_hdr_fired = ref false
      and body_hdr_fired = ref false
      and store_hdr_fired = ref false in
      Fun.protect
        ~finally:(fun () ->
          cfg := { !cfg with Emit_render.Cfg.tw = Emit_render.Cfg.Tw_default })
        (fun () ->
           cfg := { !cfg with Emit_render.Cfg.tw = Emit_render.Cfg.Tw_zsplit k.tw_off };
           List.iter
             (fun (zn : ZNode.t) ->
                if ZNode.is_store zn
                then (
                  (match k.sched with
                   | ZS_legacy ->
                     if not !store_hdr_fired
                     then (
                       store_hdr_fired := true;
                       Buffer.add_string buf store_hdr)
                   | ZS_afterdef | ZS_off -> ());
                  match zn.ZNode.node with
                  | ZStore { unit_idx; _ } ->
                    if Hashtbl.mem residual unit_idx
                    then
                      Buffer.add_string
                        buf
                        "        /* UNSUNK RESIDUAL (slot->tag table / schedule \
                         mismatch) */\n";
                    Buffer.add_string buf store_units.(unit_idx).su_text
                  | ZArith _ | ZLoad _ -> assert false)
                else (
                  match zn.ZNode.node with
                  | ZLoad { unit_idx } ->
                    (* preamble unit (ZNode.is_load; distinct from the
                        ZArith-wrapped NK_Load value copies, which are
                        body defs and render below) *)
                    assert (ZNode.is_load zn);
                    if not !load_hdr_fired
                    then (
                      load_hdr_fired := true;
                      Buffer.add_string buf !load_hdr_s);
                    Buffer.add_string buf load_units.(unit_idx)
                  | ZArith e ->
                    if not !body_hdr_fired
                    then (
                      body_hdr_fired := true;
                      Buffer.add_string buf body_hdr);
                    if
                      (not (Hashtbl.mem seen e.Ir.tag))
                      && not (Hashtbl.mem inline_set e.Ir.tag)
                    then (
                      Hashtbl.replace seen e.Ir.tag ();
                      Buffer.add_string
                        buf
                        (Emit_render.render_node_def
                           ~sc
                           ~cfg:!cfg
                           ~isa
                           ~in_place:false
                           ~t1s:false
                           ~strided:true
                           ~inline_set:(Some inline_set)
                           e);
                      Buffer.add_char buf '\n')
                  | ZStore _ -> assert false))
             combined));
    Buffer.add_string buf "    }\n"
  in
  (* ── the shared 11-arg z ABI signature + computed (void) list ── *)
  let uses stride =
    let edge_uses = function
      (* section bases are stride-scaled, so the ZTURN-S edges USE their
         stride (E_sect_tap "OLs" reads/writes at 4*(sec*OLs + k)). *)
      | E_planes s | E_z s | E_sect_tap s | E_sect_tr4 s -> s = stride
      | E_blocks -> false
    in
    edge_uses k.in_edge || edge_uses k.out_edge
  in
  let plain_voids =
    String.concat
      " "
      (List.map
         (fun p -> Printf.sprintf "(void)%s;" p)
         (List.concat
            [ ([ "zin_unused"; "zout_unused" ]
               @ if k.nat_in || k.nat_out then [] else [ "tw_im" ])
            ; (if uses "Ls" then [] else [ "Ls" ])
            ; [ "Gs" ]
            ; (if uses "OLs" then [] else [ "OLs" ])
            ; [ "OGs" ]
            ; (if k.twiddled then [] else [ "tw_re" ])
            ]))
  in
  let emit_signature () =
    (* M4 phase 3: the FROZEN z ABI from Abi.z11_signature — one source (was
       one of three byte-identical hand prints); the DERIVED silencer list
       (plain_voids) stays this family's own. *)
    Buffer.add_string
      buf
      (Abi.z11_signature ~symbol:fname ~target_attr:isa.Isa.target_attr ());
    Buffer.add_string buf (Printf.sprintf "    %s\n" plain_voids)
  in
  (* ── s0t fwd body: CLOSED-FORM TEMPLATE (the tr4_str precedent — bodies
        the col-loop IR cannot express). Every value in the 18-op turn
        lattice is an interleaved [re,im,re,im] 2-complex vector; the IR
        models re/im as separate slots per Output, and the cross-pair
        permute2f128 stores target two section pairs per iteration — no
        slot/colo decomposition exists. VERBATIM transcription of the proven
        prototype (src/core/oop/zturn_proto.h _vfft_zturn_s0t_fwd, memcmp-
        EXACT at all four cells). Source order is LOAD-BEARING (commuted-
        twins proof, 18-p5 census, gcc ordering doctrine): loads; at0..at3;
        ar; aY0, aY2, aY1, aY3 (Y2 BEFORE Y1); au0..au3; stores sec re,
        sec im, sec' re, sec' im. Prototype's noinline/used dropped at
        emitter transcription (consensus doc §6). ── *)
  let emit_s0t_body () =
    let unlo = Isa.intr isa "unpacklo_pd"
    and unhi = Isa.intr isa "unpackhi_pd"
    and p2f = Isa.intr isa "permute2f128_pd" in
    let line s = Buffer.add_string buf ("        " ^ s ^ "\n") in
    let sline s = Buffer.add_string buf ("        " ^ s ^ ";\n") in
    Buffer.add_string buf "    for (size_t k = 0; k + 4 <= count; k += 4) {\n";
    List.iter
      (fun (p, plus, sec_a, sec_b, pos_a, pos_b) ->
         let v n = p ^ n in
         Buffer.add_string
           buf
           (Printf.sprintf
              "        /* ---- half %s: positions %s, %s -> sections %d, %d (bitrev2) \
               ---- */\n"
              (String.uppercase_ascii p)
              pos_a
              pos_b
              sec_a
              sec_b);
         for l = 0 to 3 do
           line
             (Isa.const_decl
                isa
                (v (string_of_int l))
                (Isa.loadu_pd isa (leg_addr "zin" l "Ls" 0 plus)))
         done;
         line (Isa.const_decl isa (v "t0") (Isa.add_pd isa (v "0") (v "2")));
         line (Isa.const_decl isa (v "t1") (Isa.sub_pd isa (v "0") (v "2")));
         line (Isa.const_decl isa (v "t2") (Isa.add_pd isa (v "1") (v "3")));
         line (Isa.const_decl isa (v "t3") (Isa.sub_pd isa (v "1") (v "3")));
         line
           (Isa.const_decl
              isa
              (v "r")
              (Isa.xor_mask_pd isa (Isa.cflip_pd isa (v "t3")) "_zs0t_mim"));
         line (Isa.const_decl isa (v "Y0") (Isa.add_pd isa (v "t0") (v "t2")));
         line (Isa.const_decl isa (v "Y2") (Isa.sub_pd isa (v "t0") (v "t2")));
         line (Isa.const_decl isa (v "Y1") (Isa.add_pd isa (v "t1") (v "r")));
         line (Isa.const_decl isa (v "Y3") (Isa.sub_pd isa (v "t1") (v "r")));
         line
           (Isa.const_decl
              isa
              (v "u0")
              (Printf.sprintf "%s(%s, %s)" unlo (v "Y0") (v "Y1")));
         line
           (Isa.const_decl
              isa
              (v "u1")
              (Printf.sprintf "%s(%s, %s)" unhi (v "Y0") (v "Y1")));
         line
           (Isa.const_decl
              isa
              (v "u2")
              (Printf.sprintf "%s(%s, %s)" unlo (v "Y2") (v "Y3")));
         line
           (Isa.const_decl
              isa
              (v "u3")
              (Printf.sprintf "%s(%s, %s)" unhi (v "Y2") (v "Y3")));
         sline
           (Isa.storeu_pd
              isa
              (leg_addr "zout" sec_a "Ls" 0 0)
              (Printf.sprintf "%s(%s, %s, 0x20)" p2f (v "u0") (v "u2")));
         sline
           (Isa.storeu_pd
              isa
              (leg_addr "zout" sec_a "Ls" 0 vw)
              (Printf.sprintf "%s(%s, %s, 0x20)" p2f (v "u1") (v "u3")));
         sline
           (Isa.storeu_pd
              isa
              (leg_addr "zout" sec_b "Ls" 0 0)
              (Printf.sprintf "%s(%s, %s, 0x31)" p2f (v "u0") (v "u2")));
         sline
           (Isa.storeu_pd
              isa
              (leg_addr "zout" sec_b "Ls" 0 vw)
              (Printf.sprintf "%s(%s, %s, 0x31)" p2f (v "u1") (v "u3"))))
      [ "a", 0, 0, 2, "k", "k+1"; "b", vw, 1, 3, "k+2", "k+3" ];
    Buffer.add_string buf "    }\n"
  in
  if k.base = "s0t" && not k.bwd
  then (
    (* ── s0t fwd: closed-form template — no DAG, no prepare ── *)
    emit_signature ();
    emit_s0t_body ();
    Buffer.add_string buf "}\n")
  else if k.uj2
  then (
    (* ── sterm2: one function, shared k cursor, 2-instance main loop
           (SU-braided) + baseline-shaped VW-column tail. The two DAGs are
           prepared sequentially — main is rendered before the tail's
           Ir.reset (GOTCHA 2). ── *)
    emit_signature ();
    Buffer.add_string buf "    size_t k = 0;\n";
    let main_dag = prepare ~two_inst:true in
    emit_col_loop
      ~open_line:
        (Printf.sprintf "    for (; k + %d <= count; k += %d) {\n" (2 * vw) (2 * vw))
      ~ninst:2
      main_dag;
    Buffer.add_string
      buf
      (Printf.sprintf
         "    /* ---- baseline-shaped %d-column tail (count %% %d == %d) ---- */\n"
         vw
         (2 * vw)
         vw);
    let tail_dag = prepare ~two_inst:false in
    emit_col_loop
      ~open_line:(Printf.sprintf "    for (; k + %d <= count; k += %d) {\n" vw vw)
      ~ninst:1
      tail_dag;
    Buffer.add_string buf "}\n")
  else if not k.group_loop
  then (
    (* ── plain kind: exported function wraps the column loop directly ── *)
    emit_signature ();
    let dag = prepare ~two_inst:false in
    emit_col_loop
      ~open_line:
        (Printf.sprintf "    for (size_t k = 0; k + %d <= count; k += %d) {\n" vw vw)
      ~ninst:1
      dag;
    Buffer.add_string buf "}\n")
  else (
    (* ── msg: static always_inline body + thin group-loop wrapper.
           The body carries NO target attribute (always_inline requires the
           callee's target ⊆ caller's; it inlines into the attributed
           wrapper). Wrapper shape mirrors legacy codelet_zil byte-for-byte:
           in-place on zout (zin voided), bp += 2·R·Ls, twg += (R-1)·2·VW. ── *)
    let body_name = Printf.sprintf "_zsg%d%s_body" radix (if k.bwd then "b" else "f") in
    Buffer.add_string
      buf
      (Printf.sprintf
         "static __attribute__((always_inline)) inline void %s(\n\
         \    const double * __restrict__ zin, double * __restrict__ zout,\n\
         \    const double *tw_re, size_t Ls, size_t count)\n\
          {\n"
         body_name);
    let dag = prepare ~two_inst:false in
    emit_col_loop
      ~open_line:
        (Printf.sprintf "    for (size_t k = 0; k + %d <= count; k += %d) {\n" vw vw)
      ~ninst:1
      dag;
    Buffer.add_string buf "}\n\n";
    (* M4 phase 3: the driver wrapper's FROZEN z ABI also comes from
        Abi.z11_signature (the third of the three hand prints). *)
    Buffer.add_string
      buf
      (Abi.z11_signature ~symbol:fname ~target_attr:isa.Isa.target_attr ());
    Buffer.add_string
      buf
      (Printf.sprintf
         "    (void)zin; (void)zin_unused; (void)zout_unused; (void)tw_im;\n\
         \    (void)OLs; (void)OGs;\n\
         \    double *bp = zout;\n\
         \    const double *twg = tw_re;\n\
         \    for (size_t g = 0; g < Gs; g++) {\n\
         \        %s(bp, bp, twg, Ls, count);\n\
         \        bp += 2 * (size_t)%d * Ls;\n\
         \        twg += %d;\n\
         \    }\n\
          }\n"
         body_name
         radix
         ((radix - 1) * 2 * vw)));
  Buffer.contents buf
;;
