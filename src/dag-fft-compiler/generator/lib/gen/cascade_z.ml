(* cascade_z.ml — the BLOCK-SPLIT stage-kernel emitter (the Cascade_z feature
 * module, §9 #35; the file keeps its historical name, and the emitted files
 * keep the "codelet_zsplit.ml" provenance stem they have always carried).
 *
 * Emits the stage kernels of the interleaved K=1 engines whose INTERIOR is
 * split (64-B [re x4][im x4] blocks) and whose EDGES are the caller's packed
 * interleaved z:
 *
 *   ZTURN-T, natural class (docs/design/CONTRACT.md, ztt_odd_design.md):
 *     t0tp/t0tpb  ingest  — twiddle-free radix R0 on natural z legs, the turn
 *                           lattice, RUN-CONTIGUOUS stores through rb[] (E_runs)
 *     tmg/tmgb    mid     — in-place group-looped combine, column-varying
 *                           pre-twiddle stream (odd radices 3/5/7/9/15 too)
 *     tlf/tlfb    last    — the combine + REINT packed stores = natural out
 *     tlfi/tlfib  last    — tlf with its output streams prefetched (in place)
 *   ZTURN-T, plain = scrambled class (docs/design/ztt_scrambled_design.md):
 *     t0d         ingest  — DFT then POST-twiddle, block-split leg-major stores
 *     tmgd        mid     — tmg with the twiddle AFTER the butterfly
 *     tld/tldb    last    — TR4 block loads, twiddle-free DFT, unpack-only
 *                           interleaved stores IN PLACE (tldb = its mirror)
 *   il2p's odd mids (the flat DIT front door):
 *     msz/mszb/mszt       — the split-body mid between interleaved edges,
 *                           unordered lanes, odd-count narrow arms
 *
 * Every kernel is derived through the production DAG pipeline:
 *
 *   Dft.dft_expand / dft_expand_twiddled (math layer, elem-ref space)
 *     -> Pipeline.prepare_codelet (hash-cons + algsimp/FMA cascade)
 *     -> Schedule.su_schedule (SR list scheduler over the DAG)
 *     -> Emit_render.render_node_def (Isa-parameterized rendering)
 *
 * with this family's EDGES (the block-split / z / block / column-run
 * load-store shapes below) and TWIDDLE RENDERING (Emit_state.current_tw_zsplit:
 * [c x VW][s x VW] records in tw_re; the z ABI's tw_im slot carries rb[] for
 * t0tp and is dead elsewhere). The one body the DAG cannot express — the
 * ingest's turn lattice, whose values are interleaved 2-complex vectors — is a
 * closed-form template (emit_t0tp4_body / emit_t0tp8_body).
 *
 * ABI: the frozen 11-arg z ABI (Abi.z11_signature). The group-looped kinds
 * export a static always_inline BODY plus the thin wrapper that walks the Gs
 * groups; the fused ZTURN-T drivers (ztt_drivers.ml) re-emit the same bodies
 * with ~body_only:true, so a fused cell and the staged executor run identical
 * arithmetic (ztt_gate holds staged == fused bitwise).
 *
 * TIER GATE: radix 4/8 only, monolithic BY DESIGN (16 planes fit the ymm
 * file), except the MIDS, which carry no lane lattice and emit the odd
 * radices 3/5/7/9/15. Dft.should_spill is NOT consulted (its n>=5 clause
 * would put R=8 on the spill recipe).
 *
 * Bwd kinds run with ~table_conj because the runtime tables are ALREADY
 * conjugated by the create — the double-conj trap, zil_pipeline_port.md §6.1.
 * ------------------------------------------------------------------
 * MODULE CARD (codelet_zsplit.ml — grep "MODULE CARD" for the full set)
 * ROLE: DAG-pipeline emitter for the block-split stage kernels.
 * PIPELINE: Dft -> Pipeline.prepare_codelet -> Schedule.su_schedule ->
 * this family's edges + Emit_render.render_node_def.
 * PUBLIC SURFACE: emit_codelet (gen_main --zp-* flags; ztt_drivers.ml).
 * DEPS: Dft, Algsimp, Pipeline, Schedule, Emit_c (render + provenance +
 * compute_inline_set + current_tw_zsplit via the Emit_state chain), Isa,
 * Uarch.
 * GOTCHA: sets Emit_state.current_tw_zsplit for the duration of emission
 * (Fun.protect-reset); no other family may leave it non-None.
 * ------------------------------------------------------------------ *)

(* ─── kind table ──────────────────────────────────────────────────── *)

(* Edge shapes for the column loop's memory boundary:
     E_planes — block-split planes: re at 2*(l*S+k), im +VW. Plain vector
                loads/stores, shuffle-free (the interior contract).
     E_z      — natural interleaved z at the same leg addressing: two z
                vectors per leg, DEINT on load / REINT on store (the API
                boundary; shuffles paid once per transform).
     E_blocks — col-block edge: per column, R consecutive complexes as R/VW
                [re×VW][im×VW] blocks at 2·R·(k+c); TR4 register transposes
                swap column-lane ↔ leg-index (load: blocks → leg-major lanes;
                store: the inverse). tld's load side, tldb's store side.
   Each stride-using edge carries the C stride NAME ("Ls" | "OLs") —
   the ingest/mids run on Ls, the terminator's packed output on OLs. *)
type zs_edge =
  | E_planes of string
  | E_z of string
  | E_blocks
  | E_runs
    (* ZTURN-T run-contiguous ingest store (t0tp, CONTRACT.md 5 / 7.1):
       column c's R0 outputs land at PLANE complex rb[c]*R0 .. +R0-1 =
       R0/4 consecutive 64-B blocks from doubles zout + 2*R0*rb[c]; rb[]
       (one size_t per column) rides in the tw_im slot. The whole edge is
       the turn lattice's store side, template-emitted. *)
  | E_zcol
    (* ZTURN-T PLAIN (scrambled) column run, INTERLEAVED, IN PLACE
       (docs/design/ztt_scrambled_design.md): column c's R consecutive
       complexes at doubles 2*R*(k+c) -- the same span E_blocks addresses,
       in the caller's z format. Leg p's four column outputs leave as TWO
       registers by unpacklo/hi ALONE: [col k, col k+2] at 2*R*k + 8p and
       [col k+1, col k+3] at +4 -- no permute; the lane order inside the
       4-column span is part of the plan's permutation (tabulated at
       create, never read at run time). Store side = tld (the plain last),
       load side = tldb (its mirror): two loads + unpacklo/hi per leg give
       the ORDERED lanes k..k+3 back. Because the 4 columns' span is exactly
       what the E_blocks edge on the other side reads/writes, both kinds are
       in place. *)

type zs_kind =
  { base : string
    (* C-name stem: "t0tp" | "tmg" | "tlf" | "tlfi" | "t0d" | "tmgd" | "tld"
       | "msz" | "mszt" *)
  ; bwd : bool
  ; twiddled : bool (* false: n1 math + (void)tw_re (t0tp, tld) *)
  ; tw_off : string
    (* C base-offset expression for the twiddle record stream: "" for
       table-start records (msz: per-group splat-pair sets), "%TWF*(size_t)k"
       for the column-varying stream (the ZTURN-T kinds). *)
  ; in_edge : zs_edge
  ; out_edge : zs_edge
  ; tw_group_reset : bool
    (* ZTURN-T mids/last (tmg/tlf): the twiddle w_{RL}^{r*b} depends on the
       column b inside the run and NOT on the group, so ONE stream serves
       every group -- the record offset advances with k inside the body
       (tw_off = %TWF*k) and the group-loop wrapper does NOT bump twg
       (CONTRACT.md 3). *)
  ; dif : bool
    (* twiddle PLACEMENT direction for dft_expand_twiddled. false = DIT
       (pre-twiddle at Fwd). true = DIF (POST-twiddle at Fwd: t0d/tmgd, the
       plain class). Placement travels with (direction, sign) in
       dft.ml:271-277, so the backward twins that must keep the DIT order
       (PRE-twiddle conj then IDFT: tmgb/tlfb/tlfib/mszb) also set it — at
       Bwd, (DIF, Bwd) lands on PRE. *)
  ; lanes_u : bool
    (* UNORDERED PLANE LANES (2026-09-05, split-body form detail): the z
       edges leave the lanes as unpacklo/hi produce them ([0,2,1,3]) and
       skip the two 0xD8 permutes per leg per 4 columns (zu_noperm); the
       per-block record-sets are broadcast, so lane order is moot. msz is
       unordered by construction. *)
  ; narrow_arms : bool
    (* il_odd_count_tail.md §3 for the SPLIT family (2026-09-05): after the
       VW-column wide loop, the SAME scheduled DAG is re-rendered as two
       trailing arms — 2 columns at Isa.sse2 (VEX-128: unpacklo/hi of two
       1-complex loads IS the whole deinterleave) and 1 column at
       Isa.scalar (re/im loaded directly) — so count becomes ANY >= 1.
       Twiddle records stay the wide table's (Cfg.tw_vw). E_z kinds only;
       msz only — the ZTURN-T kinds keep the count % 4 == 0 form. *)
  ; prefetch_out : int
    (* IN-PLACE terminator (tlfi, 2026-09-09, zturn_t_ship_plan.md 9): the
       distance, in COLUMN QUADS, at which every output stream's line is
       prefetched (T0) at the top of each column-quad iteration — R
       prefetches of zout[2*(r*OLs + k + VW*prefetch_out)]. 0 = none. Why:
       in place the terminator's stores land in the caller's buffer, whose
       lines left L1 while the plane was the working set, and each store
       waits on its line fill (+17..25% measured); out of place the same
       stores hit lines the terminator's own loads just filled, so the
       dest driver needs nothing. Arithmetic is untouched: the in-place
       result stays bitwise the out-of-place one. Measured band 4..8 quads;
       32 overshoots the L1 set. *)
  }

let kind_of_string (s : string) : zs_kind =
  let dflt =
    { base = ""
    ; bwd = false
    ; twiddled = true
    ; tw_off = ""
    ; tw_group_reset = false
    ; in_edge = E_planes "Ls"
    ; out_edge = E_planes "Ls"
    ; dif = false
    ; lanes_u = false
    ; narrow_arms = false
    ; prefetch_out = 0
    }
  in
  match s with
  | "msz" ->
    (* the split-body mid on our contract (2026-09-05, owner: "kernel-level
       boundary IL, split body is fine"): the group-looped split mid with
       INTERLEAVED z on BOTH edges — deinterleave on load, the shuffle-free
       split body, reinterleave on store — no split planes in memory, no
       extra pass. The lane order inside is left as unpacklo/hi produce it
       ([0,2,1,3]): the two 0xD8 permutes of the ordered z edges are skipped
       (zu_noperm), which is what takes the boundary from 2.0 to 1.0
       shuffles per point; the per-block record-sets are broadcast, so lane
       order is moot. *)
    { dflt with
      base = "msz"
    ; in_edge = E_z "Ls"
    ; out_edge = E_z "Ls"
    ; lanes_u = true
    ; narrow_arms = true
    }
  | "mszb" ->
    (* the flat DIT's backward msz (2026-09-05): the CONJUGATE pipeline keeps
       the stage order, so this is PRE-twiddle (conj table, driver-side) +
       IDFT — dft.ml places (DIF, Bwd) PRE, hence dif = true here. Same
       edges, lanes and arms. *)
    { dflt with
      base = "msz"
    ; bwd = true
    ; dif = true
    ; in_edge = E_z "Ls"
    ; out_edge = E_z "Ls"
    ; lanes_u = true
    ; narrow_arms = true
    }
  | "mszt" ->
    (* msz's TRANSPOSED backward (2026-09-05): IDFT block then the
       conjugated twiddle POST (dft.ml places (DIT, Bwd) POST) — the flat
       DIT's scrambled class consumes the comb by running the stages in
       reverse, each transposed. Own stem so it links beside mszb. *)
    { dflt with
      base = "mszt"
    ; bwd = true
    ; in_edge = E_z "Ls"
    ; out_edge = E_z "Ls"
    ; lanes_u = true
    ; narrow_arms = true
    }
  | "t0tp" | "t0tpb" ->
    (* ZTURN-T INGEST (CONTRACT.md 5, 7.1): natural packed z legs at
       stride Ls = N/R0, twiddle-free radix R0 (the ingest's own radix IS
       R0 -- no _r0 tag), the turn lattice, stores run-contiguous through
       rb[] (E_runs). The packed-through form the sub-2048 campaign raced
       in ("t0tp": 1.42-1.78x over the split-butterfly t0t). bwd = the same
       leaf with conjugate roots (the +i mask): the inverse pipeline keeps
       the DIT order, natural in and out, tables conjugated by the create. *)
    { dflt with
      base = "t0tp"
    ; bwd = s = "t0tpb"
    ; twiddled = false
    ; in_edge = E_z "Ls"
    ; out_edge = E_runs
    }
  | "tmg" | "tmgb" ->
    (* ZTURN-T MID (CONTRACT.md 3, 7.2): the group-looped in-place block
       combine with the COLUMN-VARYING pre-twiddle: records advance with k
       inside the body ((R-1) records of [c x4][s x4] per column quad) and
       the cursor RESETS per group. *)
    { dflt with
      base = "tmg"
    ; bwd = s = "tmgb"
      (* the inverse keeps the DIT ORDER: PRE-twiddle (conj, table_conj) then
         the IDFT butterfly. Placement travels with (direction, sign) in
         dft.ml, so (DIT, Bwd) would be IDFT-then-POST (the transposed
         pipeline); dif = true at Bwd lands on PRE -- mszb's rule. *)
    ; dif = s = "tmgb"
    ; tw_off = "%TWF*(size_t)k"
    ; tw_group_reset = true
    }
  | "tlf" | "tlfb" ->
    (* ZTURN-T LAST (CONTRACT.md 6, 7.3): tmg's combine with the REINT
       packed store at leg*OLs + k -- natural interleaved output, no
       reordering pass anywhere. Group-looped over distinct in/out
       pointers (Gs = 1 at the terminal stage of a K-stage chain). Alias-
       tolerant: with Ls == OLs an in-place call rewrites exactly the
       blocks it read (zt_last.c, "ALIASING"), which is what lets the
       fused driver run the whole pipeline in the destination. *)
    { dflt with
      base = "tlf"
    ; bwd = s = "tlfb"
    ; dif = s = "tlfb"   (* PRE-twiddle at Bwd, as tmgb *)
    ; tw_off = "%TWF*(size_t)k"
    ; tw_group_reset = true
    ; out_edge = E_z "OLs"
    }
  | "tlfi" | "tlfib" ->
    (* ZTURN-T IN-PLACE LAST (2026-09-09, zturn_t_ship_plan.md 9): tlf with
       its OUTPUT STREAMS PREFETCHED four column quads ahead — the one thing
       that separates the placements' terminators (see prefetch_out). Bound
       by the `plane` drivers; the `dest` drivers keep tlf. Same arithmetic,
       same edges, same stream: in place bitwise the out-of-place result. *)
    { dflt with
      base = "tlfi"
    ; bwd = s = "tlfib"
    ; dif = s = "tlfib"   (* PRE-twiddle at Bwd, as tmgb *)
    ; tw_off = "%TWF*(size_t)k"
    ; tw_group_reset = true
    ; out_edge = E_z "OLs"
    ; prefetch_out = 4
    }
  (* ── ZTURN-T PLAIN = the scrambled class (docs/design/ztt_scrambled_design.md):
        Sande-Tukey in place on the chain. Stage 0 t0d (interleaved legs at
        stride Ls = N/R0, DFT, POST-twiddle, block-split LEG-MAJOR stores at
        2*(p*Ls + k): R0 sequential streams, no rb[]); mids tmgd (tmg with the
        twiddle AFTER the butterfly); last tld (adjacent legs, twiddle-free,
        interleaved stores IN PLACE). The backward is the stage-by-stage
        inverse: tldb (tld's mirror) then the SHIPPED tmgb (PRE conj + IDFT)
        and tlfb (PRE conj + IDFT, natural interleaved out). No table, no
        plane, no buffer-mode axis. ── *)
  | "t0d" ->
    { dflt with
      base = "t0d"
    ; dif = true
    ; tw_off = "%TWF*(size_t)k"
    ; tw_group_reset = true
    ; in_edge = E_z "Ls"
    ; out_edge = E_planes "Ls"
      (* Two twins were MEASURED here 2026-09-14 (zt_scr_spike_results.md)
         and REFUTED, do not re-try: prefetch_out = 4 (the tlfi mechanism) —
         no gain out of place, none in place; and a packed-w^1 ingest with
         the powers from a squaring tree (the stream 7x smaller) — correct
         and 12..48% SLOWER at every L2-resident cell. The stage is bound by
         neither store-miss latency nor its twiddle bytes. *)
    }
  | "tmgd" ->
    (* tmg's dataflow with DIF placement (dft.ml: (DIF, Fwd) lands on POST) *)
    { dflt with
      base = "tmgd"
    ; dif = true
    ; tw_off = "%TWF*(size_t)k"
    ; tw_group_reset = true
    }
  | "tld" ->
    { dflt with base = "tld"; twiddled = false; in_edge = E_blocks; out_edge = E_zcol }
  | "tldb" ->
    { dflt with
      base = "tld"
    ; bwd = true
    ; twiddled = false
    ; in_edge = E_zcol
    ; out_edge = E_blocks
    }
  | other ->
    failwith
      (Printf.sprintf
         "codelet_zsplit: unknown kind %s (supported: msz mszb mszt t0tp t0tpb tmg tmgb \
          tlf tlfb tlfi tlfib t0d tmgd tld tldb)"
         other)
;;

(* ─── emission ────────────────────────────────────────────────────── *)

(* msz: the z edges skip the lane-order permutes (see kind_of_string). Set
   per emission from the kind; gen_set runs many cells in one process. *)
(* "%TWF" in a kind's tw_off = the loaded-stream group pitch, 2*(R-1)
   doubles per column ((R-1) records of 2*VW doubles per 4-column group),
   resolved at emission where the radix is known. *)
let resolve_tw_off (radix : int) (s : string) : string =
  let tag = "%TWF" in
  let n = String.length s
  and m = String.length tag in
  let b = Buffer.create n in
  let i = ref 0 in
  while !i < n do
    if !i + m <= n && String.sub s !i m = tag
    then (
      Buffer.add_string b (string_of_int (2 * (radix - 1)));
      i := !i + m)
    else (
      Buffer.add_char b s.[!i];
      incr i)
  done;
  Buffer.contents b
;;

let zu_noperm = ref false

(* ZTURN-T body names (the probe's: zt_ingest2.c / zt_mid.c / zt_last.c):
   the fused driver TU (ztt_drivers.ml) re-emits these bodies beside the
   per-kind codelets, so the name is shared here. *)
let ztt_body_name ~(base : string) ~(radix : int) ~(bwd : bool) : string =
  Printf.sprintf "_z%s%d%s_body" base radix (if bwd then "b" else "f")
;;

let emit_codelet
      ~(body_only : bool)
      ~store_on_compute
      ~(kind : string)
      ~(radix : int)
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
  zu_noperm := k.lanes_u;
  if radix <> 4 && radix <> 8
     && not
          ((k.base = "msz" || k.base = "mszt" || k.base = "tmg" || k.base = "tmgd")
           && List.mem radix [ 3; 5; 7; 9; 15 ])
  then
    failwith
      "codelet_zsplit: split family is radix 4/8 only (see TIER GATE) — except the \
       MIDS (msz/mszt; ZTURN-T's tmg/tmgb/tmgd since 2026-09-14, ztt_odd_design.md; \
       t0tp/tlf/tld are lane lattices and stay 4/8), which carry no lane geometry \
       and emit odd radices 3/5/7/9/15";
  let vw = isa.Isa.vec_width in
  if vw <> 4
  then
    (* The generator side is width-parameterized, but the RUNTIME block
       geometry ([re×VW][im×VW]) is baked into the plan builders at VW=4. *)
    failwith
      "codelet_zsplit: runtime block geometry is VW=4 until zsplit.h is parameterized";
  let dir_s = if k.bwd then "bwd" else "fwd" in
  let fname = Printf.sprintf "radix%d_z_%s_%s_%s" radix k.base dir_s isa.Isa.name in
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
     Called once per emitted body. Twiddled kinds: dft_expand_twiddled with
     the flat splat-pair records — (DIT, Fwd) = pre-twiddle (tmg/tlf/msz),
     (DIF, Fwd) = post-twiddle (t0d/tmgd) — and at Bwd ~table_conj because
     the create hands the kernels an already-conjugated table. Twiddle-free
     kinds (t0tp/tld): dft_expand. *)
  let prepare ()
    : (Expr.elem_ref * Ir.t) list
      * (Expr.elem_ref option * Ir.t) list
      * (int, unit) Hashtbl.t
    =
    let raw_assigns =
      if k.twiddled
      then
        Dft.dft_expand_twiddled
          ~policy:Dft.TP_Flat
          ~direction:(if k.dif then Dft.DIF else Dft.DIT)
          ~sign
          ~table_conj:k.bwd
          radix
      else Dft.dft_expand ~sign radix
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
       \ * CONTRACT: %s.\n\
       \ * %s%s */\n"
       vw
       vw
       vw
       (match k.base, k.bwd with
        | "t0tp", b ->
          Printf.sprintf
            "t0tp (ZTURN-T ingest: natural packed z legs at stride N/R0, twiddle-free \
             radix-%d, s0t's turn lattice, RUN-CONTIGUOUS block stores at plane complex \
             rb[c]*R0 through the tw_im-carried run-base table), %s."
            radix
            (if b then "bwd (conjugate roots)" else "fwd")
        | "tmg", b ->
          Printf.sprintf
            "tmg (ZTURN-T mid: msg's in-place group-looped combine with the COLUMN-VARYING \
             pre-twiddle w_RL^(r*b): (R-1) records per column quad advancing with k, \
             cursor RESET per group), %s."
            (if b then "bwd (table conjugated by the create)" else "fwd")
        | "tlf", b ->
          Printf.sprintf
            "tlf (ZTURN-T last: tmg's combine + REINT packed stores at leg*OLs + k = \
             natural interleaved out, group-looped over distinct in/out pointers), %s."
            (if b then "bwd (table conjugated by the create)" else "fwd")
        | "tlfi", b ->
          Printf.sprintf
            "tlfi (ZTURN-T IN-PLACE last: tlf with every output stream prefetched 4 \
             column quads ahead — the caller's buffer went cold under the plane; same \
             arithmetic, bitwise tlf), %s."
            (if b then "bwd (table conjugated by the create)" else "fwd")
        | "msz", false ->
          "msz (the msg body between interleaved edges, unordered lanes, il_odd_count_tail §3 \
           arms; per-block splat records), fwd."
        | "msz", true ->
          "mszb (msz's backward: the conjugate pipeline's PRE-twiddle conj + IDFT, dif=true), \
           bwd."
        | "mszt", true ->
          "mszt (msz's TRANSPOSED backward: IDFT + POST-twiddle conj — the scrambled class), \
           bwd."
        | "t0d", false ->
          "t0d (ZTURN-T PLAIN ingest, the scrambled class: natural packed z legs at \
           stride Ls = N/R0, DEINT, radix-R0 DFT, POST-twiddle w_N^(p*b) as tlf's \
           column-quad stream, block-split LEG-MAJOR stores at 2*(p*Ls + k) -- R0 \
           sequential streams, no rb[]; in place when zin == zout; every output \
           stream prefetched 4 column quads ahead, the tlfi mechanism), fwd."
        | "tmgd", false ->
          "tmgd (ZTURN-T PLAIN mid = tmg with the twiddle AFTER the butterfly: in-place \
           group-looped combine, POST-twiddle w_Len^(p*b), cursor reset per group), fwd."
        | "tld", false ->
          "tld (ZTURN-T PLAIN last, IN PLACE: TR4 loads of R adjacent block-split \
           complexes per column, twiddle-free radix-R DFT, unpack-only interleaved \
           stores back into the same span = the scrambled output), fwd."
        | "tld", true ->
          "tldb (ZTURN-T PLAIN backward ingest, IN PLACE = tld's mirror: unpack-only \
           interleaved loads of the scrambled input, twiddle-free IDFT, TR4 block-split \
           stores into the same span), bwd."
        | b, d ->
          failwith
            (Printf.sprintf
               "codelet_zsplit: no header text for kind %s %s"
               b
               (if d then "bwd" else "fwd")))
       (if k.narrow_arms
        then
          Printf.sprintf
            "count: ANY >= 1 (%d-column wide loop, then 2-column VEX-128 and 1-column scalar arms — il_odd_count_tail.md §3)"
            vw
        else Printf.sprintf "count %% %d == 0 (%d columns per iteration)" vw vw)
       (if not k.twiddled
        then "tw_re/tw_im unused (twiddle-free leaf)."
        else
          Printf.sprintf
            "tw_re = %s: legs 1..R-1, %d doubles/leg [c×%d][s×%d]. tw_im unused."
            (if k.tw_group_reset
             then "(R-1) records of [c x4][s x4] PER COLUMN QUAD at %TWF*k, ONE stream for every group (cursor resets per group)"
             else "Gs per-group splat-pair sets, in-kernel cursor (twg bump/group)")
            (2 * vw)
            vw
            vw)
       (if k.bwd then " Roundtrip = N*x (no 1/N in-kernel)." else ""));
  Buffer.add_string
    buf
    (Emit_render.provenance_block
       ~family:"zsplit-pipeline"
       [ Printf.sprintf "kind=%s radix=%d dir=%s isa=%s" k.base radix dir_s isa.Isa.name
       ; (if k.twiddled
          then
            Printf.sprintf
              "math: Dft.dft_expand_twiddled TP_Flat %s%s"
              (if k.dif then "DIF" else "DIT")
              (if k.bwd then " sign=Bwd table_conj=true" else " sign=Fwd")
          else "math: Dft.dft_expand (n1)" ^ if k.bwd then " sign=Bwd" else " sign=Fwd")
       ; "prepare: Pipeline.prepare_codelet (monolithic, fuse=0)"
       ; "schedule: Schedule.su_schedule (SR list scheduler)"
       ]);
  Buffer.add_string buf "#include <immintrin.h>\n#include <stddef.h>\n\n";
  (* t0tp: the turn lattice needs the (im,re)-swap sign mask; the quarter-turn
     mask picks the sign (x(-i) fwd = im_mask, x(+i) bwd = re_mask). File-scope
     const so gcc hoists ONE load out of the loop. *)
  if k.base = "t0tp" && not body_only
  then
    Buffer.add_string
      buf
      ((if k.bwd then Isa.re_mask_decl isa "_zs0t_pim" else Isa.im_mask_decl isa "_zs0t_mim")
       ^ "\n\n");
  if k.base = "t0tp" && radix = 8 && not body_only
  then
    Buffer.add_string
      buf
      "static const __m256d _zs0t_rh = { 0.70710678118654752440, 0.70710678118654752440, \
       0.70710678118654752440, 0.70710678118654752440 };  /* 1/sqrt2: |W8^1| */\n\n";
  let body_start = ref 0 in
  (* TR4 rendering helper (E_blocks): 4 unpacks + 4 permute2f128 turning
     four column vectors into four leg/index vectors (or back). srcs/dsts
     are C variable names; dsts are declared const. *)
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
  (* leg-addressed edge address: 2*(leg*STRIDE + k), im/hi +VW *)
  let leg_addr (buf_name : string) (leg : int) (stride : string) (plus : int) : string =
    let base =
      if leg = 0
      then "2*(size_t)k"
      else Printf.sprintf "2*((size_t)%d*%s + k)" leg stride
    in
    if plus = 0
    then Printf.sprintf "%s[%s]" buf_name base
    else Printf.sprintf "%s[%s + %d]" buf_name base plus
  in
  (* ── column loop: load edge + SU-scheduled body + store edge for ONE
        prepared DAG (radix slots = the VW columns k..k+VW-1). open_line is
        the C for-statement (the narrow arms share one function-scope k). ── *)
  let emit_col_loop
        ?(nisa : Isa.t option)
        ~(open_line : string)
        ((assigns, scheduled, inline_set) :
          (Expr.elem_ref * Ir.t) list
          * (Expr.elem_ref option * Ir.t) list
          * (int, unit) Hashtbl.t)
    : unit
    =
    (* nisa: render this loop at a NARROWER ISA (the §3 arms). isa/vw are
       shadowed for the whole loop body; the twiddle records stay the wide
       table's through Cfg.tw_vw = wide_vw. *)
    let wide_vw = vw in
    let isa =
      match nisa with
      | Some i -> i
      | None -> isa
    in
    let vw = isa.Isa.vec_width in
    Buffer.add_string buf open_line;
    (* tlfi: R output-stream prefetches per column quad, prefetch_out quads
       ahead (in the WIDE loop's columns: wide_vw * prefetch_out) *)
    if k.prefetch_out > 0
    then (
      let ostride =
        match k.out_edge with
        | E_planes s | E_z s -> s
        | E_blocks | E_runs | E_zcol -> "OLs"
      in
      for r = 0 to radix - 1 do
        Buffer.add_string
          buf
          (Printf.sprintf
             "        _mm_prefetch((const char *)&zout[2*((size_t)%d*%s + k + %d)], _MM_HINT_T0);\n"
             r
             ostride
             (wide_vw * k.prefetch_out))
      done);
    (match k.in_edge with
     | E_planes s ->
       (* ── ZBlockSplit load edge: lane_{re,im}_l from the split planes.
             Leg l's re half at zin + 2*(l*S + k), im half +VW. ── *)
       Buffer.add_string buf "        /* ZBlockSplit load edge */\n";
       for leg = 0 to radix - 1 do
         Buffer.add_string
           buf
           (Printf.sprintf
              "        %s\n        %s\n"
              (Isa.const_decl
                 isa
                 (Printf.sprintf "lane_re_%d" leg)
                 (Isa.loadu_pd isa (leg_addr "zin" leg s 0)))
              (Isa.const_decl
                 isa
                 (Printf.sprintf "lane_im_%d" leg)
                 (Isa.loadu_pd isa (leg_addr "zin" leg s vw))))
       done
     | E_z s ->
       (* ── Z load edge (DEINT): two z vectors per leg, deinterleaved into
             the lane planes — unpacklo/hi + permute4x64 0xD8, the shuffles
             paid once at the API boundary. ── *)
       Buffer.add_string buf "        /* Z load edge (DEINT) */\n";
       let unlo = Isa.intr isa "unpacklo_pd"
       and unhi = Isa.intr isa "unpackhi_pd"
       and p44 = Isa.intr isa "permute4x64_pd" in
       for leg = 0 to radix - 1 do
         Buffer.add_string
           buf
           (Printf.sprintf
              "        %s\n        %s\n        %s\n        %s\n"
              (Isa.const_decl
                 isa
                 (Printf.sprintf "_zl_%d" leg)
                 (Isa.loadu_pd isa (leg_addr "zin" leg s 0)))
              (Isa.const_decl
                 isa
                 (Printf.sprintf "_zh_%d" leg)
                 (Isa.loadu_pd isa (leg_addr "zin" leg s vw)))
              (Isa.const_decl
                 isa
                 (Printf.sprintf "lane_re_%d" leg)
                 (if vw = 1
                  then Printf.sprintf "_zl_%d" leg (* scalar arm: re loaded directly *)
                  else if !zu_noperm
                  then Printf.sprintf "%s(_zl_%d, _zh_%d)" unlo leg leg
                  else Printf.sprintf "%s(%s(_zl_%d, _zh_%d), 0xD8)" p44 unlo leg leg))
              (Isa.const_decl
                 isa
                 (Printf.sprintf "lane_im_%d" leg)
                 (if vw = 1
                  then Printf.sprintf "_zh_%d" leg (* scalar arm: im = the +1 double *)
                  else if !zu_noperm
                  then Printf.sprintf "%s(_zl_%d, _zh_%d)" unhi leg leg
                  else Printf.sprintf "%s(%s(_zl_%d, _zh_%d), 0xD8)" p44 unhi leg leg)))
       done
     | E_blocks ->
       (* ── Block load edge (TR4): per column, load the R/VW block halves
             and TR4 each quad into leg-major lanes (lane = column). ── *)
       Buffer.add_string buf "        /* Block load edge (TR4) */\n";
       let halves = radix / vw in
       for c = 0 to vw - 1 do
         for h = 0 to halves - 1 do
           Buffer.add_string
             buf
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
       for h = 0 to halves - 1 do
         Buffer.add_string
           buf
           (tr4_str
              ~qid:(Printf.sprintf "lr%d_0" h)
              (Array.init 4 (fun j -> Printf.sprintf "_br%d_%d" h j))
              (Array.init 4 (fun j -> Printf.sprintf "lane_re_%d" ((h * vw) + j))));
         Buffer.add_string
           buf
           (tr4_str
              ~qid:(Printf.sprintf "li%d_0" h)
              (Array.init 4 (fun j -> Printf.sprintf "_bi%d_%d" h j))
              (Array.init 4 (fun j -> Printf.sprintf "lane_im_%d" ((h * vw) + j))))
       done
     | E_zcol ->
       (* ── Column-run z load edge (tldb, the plain backward ingest): column
             c's R consecutive INTERLEAVED complexes at 2*R*(k+c), laid out by
             tld as leg-pair registers -- leg p's pair at doubles 2*R*k + 8p:
             [col k, col k+2] then [col k+1, col k+3]. Two loads + unpacklo/hi
             per leg give the ORDERED lanes k..k+3 back; no permute. ── *)
       Buffer.add_string buf "        /* Column-run z load edge (leg-pair unpack, no permute) */\n";
       if vw <> 4 then failwith "codelet_zsplit: E_zcol assumes VW = 4";
       let unlo = Isa.intr isa "unpacklo_pd"
       and unhi = Isa.intr isa "unpackhi_pd" in
       for p = 0 to radix - 1 do
         Buffer.add_string
           buf
           (Printf.sprintf
              "        %s\n        %s\n        %s\n        %s\n"
              (Isa.const_decl
                 isa
                 (Printf.sprintf "_zl_%d" p)
                 (Isa.loadu_pd isa (blk_addr "zin" 0 (2 * vw * p))))
              (Isa.const_decl
                 isa
                 (Printf.sprintf "_zh_%d" p)
                 (Isa.loadu_pd isa (blk_addr "zin" 0 ((2 * vw * p) + vw))))
              (Isa.const_decl
                 isa
                 (Printf.sprintf "lane_re_%d" p)
                 (Printf.sprintf "%s(_zl_%d, _zh_%d)" unlo p p))
              (Isa.const_decl
                 isa
                 (Printf.sprintf "lane_im_%d" p)
                 (Printf.sprintf "%s(_zl_%d, _zh_%d)" unhi p p)))
       done
     | E_runs -> failwith "codelet_zsplit: E_runs is a store-only edge (t0tp)");
    (* per-slot output tag arrays (all edge shapes consume pairs) *)
    let re_tag = Array.make radix (-1)
    and im_tag = Array.make radix (-1) in
    List.iter
      (fun (lhs, (e : Ir.t)) ->
         match lhs with
         | Expr.Output (l, true) -> re_tag.(l) <- e.Ir.tag
         | Expr.Output (l, false) -> im_tag.(l) <- e.Ir.tag
         | _ -> failwith "codelet_zsplit: assign LHS is not Output")
      assigns;
    (* ── store edge, pre-rendered per unit (a slot, or a TR4 quad) and
          written after the body walk ── *)
    let store_hdr, (store_units : string array) =
      match k.out_edge with
      | E_planes s ->
        ( "        /* ZBlockSplit store edge */\n"
        , Array.init radix (fun leg ->
            Printf.sprintf
              "        %s;\n        %s;\n"
              (Isa.storeu_pd
                 isa
                 (leg_addr "zout" leg s 0)
                 (Printf.sprintf "t%d" re_tag.(leg)))
              (Isa.storeu_pd
                 isa
                 (leg_addr "zout" leg s vw)
                 (Printf.sprintf "t%d" im_tag.(leg)))) )
      | E_z s ->
        (* ── Z store edge (REINT): permute4x64 0xD8 each plane, then
              unpacklo/hi re-interleaves back to natural z. For the
              terminator this addressing (leg-major on OLs = N/R) IS the
              natural output. ── *)
        let unlo = Isa.intr isa "unpacklo_pd"
        and unhi = Isa.intr isa "unpackhi_pd"
        and p44 = Isa.intr isa "permute4x64_pd" in
        ( "        /* Z store edge (REINT) */\n"
        , Array.init radix (fun leg ->
            Printf.sprintf
              "        %s\n        %s\n        %s;\n        %s;\n"
              (Isa.const_decl
                 isa
                 (Printf.sprintf "_pr_%d" leg)
                 (if !zu_noperm
                  then Printf.sprintf "t%d" re_tag.(leg)
                  else Printf.sprintf "%s(t%d, 0xD8)" p44 re_tag.(leg)))
              (Isa.const_decl
                 isa
                 (Printf.sprintf "_qi_%d" leg)
                 (if !zu_noperm
                  then Printf.sprintf "t%d" im_tag.(leg)
                  else Printf.sprintf "%s(t%d, 0xD8)" p44 im_tag.(leg)))
              (Isa.storeu_pd
                 isa
                 (leg_addr "zout" leg s 0)
                 (if vw = 1
                  then Printf.sprintf "_pr_%d" leg
                  else Printf.sprintf "%s(_pr_%d, _qi_%d)" unlo leg leg))
              (Isa.storeu_pd
                 isa
                 (leg_addr "zout" leg s vw)
                 (if vw = 1
                  then Printf.sprintf "_qi_%d" leg
                  else Printf.sprintf "%s(_pr_%d, _qi_%d)" unhi leg leg))) )
      | E_blocks ->
        (* ── Block store edge (TR4 back): leg-major result vectors
              transpose to column vectors, stored as each column's R/VW
              block halves (tldb's output side). Unit = one half h: its two
              TR4 networks + their 2·VW stores. ── *)
        let halves = radix / vw in
        ( "        /* Block store edge (TR4) */\n"
        , Array.init halves (fun h ->
            let slot j = (h * vw) + j in
            let sr =
              tr4_str
                ~qid:(Printf.sprintf "sr%d_0" h)
                (Array.init 4 (fun j -> Printf.sprintf "t%d" re_tag.(slot j)))
                (Array.init 4 (fun j -> Printf.sprintf "_cr%d_%d" h j))
            and si =
              tr4_str
                ~qid:(Printf.sprintf "si%d_0" h)
                (Array.init 4 (fun j -> Printf.sprintf "t%d" im_tag.(slot j)))
                (Array.init 4 (fun j -> Printf.sprintf "_ci%d_%d" h j))
            in
            let stores =
              String.concat
                ""
                (List.init vw (fun c ->
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
            sr ^ si ^ stores) )
      | E_zcol ->
        (* ── Column-run z store edge (tld, the plain last, IN PLACE): the
              4 columns' span [2*R*k, 2*R*k + 8R) doubles is exactly what the
              E_blocks load edge read, so the stage is in place. Leg p's four
              column outputs leave as TWO interleaved registers by unpacklo/hi
              alone -- [col k, col k+2] at 2*R*k + 8p, [col k+1, col k+3] at
              +4 -- no permute; the lane order inside the span is part of the
              plan's permutation. Unit = one slot (leg), as E_z. ── *)
        let unlo = Isa.intr isa "unpacklo_pd"
        and unhi = Isa.intr isa "unpackhi_pd" in
        if vw <> 4 then failwith "codelet_zsplit: E_zcol assumes VW = 4";
        ( "        /* Column-run z store edge (leg-pair unpack, in place, no permute) */\n"
        , Array.init radix (fun sl ->
            Printf.sprintf
              "        %s;\n        %s;\n"
              (Isa.storeu_pd
                 isa
                 (blk_addr "zout" 0 (2 * vw * sl))
                 (Printf.sprintf "%s(t%d, t%d)" unlo re_tag.(sl) im_tag.(sl)))
              (Isa.storeu_pd
                 isa
                 (blk_addr "zout" 0 ((2 * vw * sl) + vw))
                 (Printf.sprintf "%s(t%d, t%d)" unhi re_tag.(sl) im_tag.(sl)))) )
      | E_runs ->
        failwith "codelet_zsplit: E_runs is template-emitted (t0tp's turn lattice)"
    in
    (* ── SU-scheduled body. Defs in schedule order (first occurrence per
          tag); single-use tags render inline at their consumer. Twiddle
          loads render via the zsplit record mode. ── *)
    Buffer.add_string buf "        /* SU-scheduled body (pipeline) */\n";
    Fun.protect
      ~finally:(fun () ->
        cfg := { !cfg with Emit_render.Cfg.tw = Emit_render.Cfg.Tw_default })
      (fun () ->
         cfg :=
           { !cfg with
             Emit_render.Cfg.tw = Emit_render.Cfg.Tw_zsplit (resolve_tw_off radix k.tw_off)
           ; Emit_render.Cfg.tw_vw = wide_vw
           };
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
                Buffer.add_char buf '\n'))
           scheduled);
    (* trailing store edge = the unit table in order *)
    Buffer.add_string buf store_hdr;
    Array.iter (Buffer.add_string buf) store_units;
    Buffer.add_string buf "    }\n"
  in
  (* ── the shared 11-arg z ABI signature + computed (void) list ── *)
  let uses stride =
    let edge_uses = function
      | E_planes s | E_z s -> s = stride
      | E_blocks | E_runs | E_zcol -> false
    in
    edge_uses k.in_edge || edge_uses k.out_edge
  in
  let plain_voids =
    String.concat
      " "
      (List.map
         (fun p -> Printf.sprintf "(void)%s;" p)
         (List.concat
            [ ([ "zin_unused"; "zout_unused" ] @ if k.base = "t0tp" then [] else [ "tw_im" ])
            ; (if uses "Ls" then [] else [ "Ls" ])
            ; [ "Gs" ]
            ; (if uses "OLs" then [] else [ "OLs" ])
            ; [ "OGs" ]
            ; (if k.twiddled then [] else [ "tw_re" ])
            ]))
  in
  (* ── t0tp's turn lattice: CLOSED-FORM TEMPLATE (the tr4_str precedent —
        a body the col-loop IR cannot express). Every value in the 18-op
        lattice is an interleaved [re,im,re,im] 2-complex vector; the IR
        models re/im as separate slots per Output, and the cross-pair
        permute2f128 stores target two runs per iteration — no slot
        decomposition exists. Source order is LOAD-BEARING (commuted-twins
        proof, 18-p5 census, gcc ordering doctrine): loads; t0..t3; r; Y0,
        Y2, Y1, Y3 (Y2 BEFORE Y1); u0..u3; the two run bases; stores. ── *)
  let s0t_mask = if k.bwd then "_zs0t_pim" else "_zs0t_mim" in
  let emit_t0tp4_body () =
    let unlo = Isa.intr isa "unpacklo_pd"
    and unhi = Isa.intr isa "unpackhi_pd"
    and p2f = Isa.intr isa "permute2f128_pd" in
    let line s = Buffer.add_string buf ("        " ^ s ^ "\n") in
    let sline s = Buffer.add_string buf ("        " ^ s ^ ";\n") in
    Buffer.add_string buf "    for (size_t k = 0; k + 4 <= count; k += 4) {\n";
    List.iter
      (fun (p, plus, pos_a, pos_b) ->
         let v n = p ^ n in
         (* E_runs: the two columns' run bases, one scalar rb[] load each *)
         let st half off = Printf.sprintf "%s[%d]" (v half) off in
         Buffer.add_string
           buf
           (Printf.sprintf
              "        /* ---- half %s: columns %s, %s -> runs rb[%s], rb[%s] ---- */\n"
              (String.uppercase_ascii p)
              pos_a
              pos_b
              pos_a
              pos_b);
         for l = 0 to 3 do
           line
             (Isa.const_decl
                isa
                (v (string_of_int l))
                (Isa.loadu_pd isa (leg_addr "zin" l "Ls" plus)))
         done;
         line (Isa.const_decl isa (v "t0") (Isa.add_pd isa (v "0") (v "2")));
         line (Isa.const_decl isa (v "t1") (Isa.sub_pd isa (v "0") (v "2")));
         line (Isa.const_decl isa (v "t2") (Isa.add_pd isa (v "1") (v "3")));
         line (Isa.const_decl isa (v "t3") (Isa.sub_pd isa (v "1") (v "3")));
         line
           (Isa.const_decl
              isa
              (v "r")
              (Isa.xor_mask_pd isa (Isa.cflip_pd isa (v "t3")) s0t_mask));
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
         line
           (Printf.sprintf "double * __restrict__ %s = zout + %d*rb[%s];" (v "pl") (2 * radix) pos_a);
         line
           (Printf.sprintf "double * __restrict__ %s = zout + %d*rb[%s];" (v "ph") (2 * radix) pos_b);
         sline
           (Isa.storeu_pd
              isa
              (st "pl" 0)
              (Printf.sprintf "%s(%s, %s, 0x20)" p2f (v "u0") (v "u2")));
         sline
           (Isa.storeu_pd
              isa
              (st "pl" vw)
              (Printf.sprintf "%s(%s, %s, 0x20)" p2f (v "u1") (v "u3")));
         sline
           (Isa.storeu_pd
              isa
              (st "ph" 0)
              (Printf.sprintf "%s(%s, %s, 0x31)" p2f (v "u0") (v "u2")));
         sline
           (Isa.storeu_pd
              isa
              (st "ph" vw)
              (Printf.sprintf "%s(%s, %s, 0x31)" p2f (v "u1") (v "u3"))))
      [ "a", 0, "k", "k+1"; "b", vw, "k+2", "k+3" ];
    Buffer.add_string buf "    }\n"
  in
  let emit_t0tp8_body () =
    (* ── the ingest at RADIX 8: legs a0..a7 at stride Ls = N/8; the radix-8
          DIF as two radix-4 butterflies over b_m = a_m + a_{m+4} (even
          outputs) and c_m = (a_m - a_{m+4}) W8^m (odd outputs), the W8
          powers as the (-i) mask (x(-i) = cflip + sign) and ONE 1/sqrt2
          multiply; then the radix-4 turn lattice TWICE per half: digits
          0..3 to block 0 of the column's run, digits 4..7 to block 1.
          Output digit d = 2e (even) / 2e+1 (odd) with e the radix-4 output
          of the b / c butterfly. ── *)
    let unlo = Isa.intr isa "unpacklo_pd"
    and unhi = Isa.intr isa "unpackhi_pd"
    and p2f = Isa.intr isa "permute2f128_pd" in
    let line s = Buffer.add_string buf ("        " ^ s ^ "\n") in
    let sline s = Buffer.add_string buf ("        " ^ s ^ ";\n") in
    Buffer.add_string buf "    for (size_t k = 0; k + 4 <= count; k += 4) {\n";
    List.iter
      (fun (p, plus, pos_a, pos_b) ->
         let v n = p ^ n in
         let r4 (i0, i1, i2, i3) (o0, o1, o2, o3) =
           (* the radix-4 DIF butterfly in the lattice's source order *)
           line (Isa.const_decl isa (v (o0 ^ "t0")) (Isa.add_pd isa (v i0) (v i2)));
           line (Isa.const_decl isa (v (o0 ^ "t1")) (Isa.sub_pd isa (v i0) (v i2)));
           line (Isa.const_decl isa (v (o0 ^ "t2")) (Isa.add_pd isa (v i1) (v i3)));
           line (Isa.const_decl isa (v (o0 ^ "t3")) (Isa.sub_pd isa (v i1) (v i3)));
           line
             (Isa.const_decl
                isa
                (v (o0 ^ "r"))
                (Isa.xor_mask_pd isa (Isa.cflip_pd isa (v (o0 ^ "t3"))) s0t_mask));
           line (Isa.const_decl isa (v o0) (Isa.add_pd isa (v (o0 ^ "t0")) (v (o0 ^ "t2"))));
           line (Isa.const_decl isa (v o2) (Isa.sub_pd isa (v (o0 ^ "t0")) (v (o0 ^ "t2"))));
           line (Isa.const_decl isa (v o1) (Isa.add_pd isa (v (o0 ^ "t1")) (v (o0 ^ "r"))));
           line (Isa.const_decl isa (v o3) (Isa.sub_pd isa (v (o0 ^ "t1")) (v (o0 ^ "r"))))
         in
         (* E_runs: quartet q (digits 4q..4q+3) = block q of the column's run,
            doubles 2*R0*rb[c] + 8q (+0 re, +VW im) *)
         let st half off = Printf.sprintf "%s[%d]" (v half) off in
         let turn (ya, yb, yc, yd) qoff tag =
           line (Isa.const_decl isa (v (tag ^ "u0")) (Printf.sprintf "%s(%s, %s)" unlo (v ya) (v yb)));
           line (Isa.const_decl isa (v (tag ^ "u1")) (Printf.sprintf "%s(%s, %s)" unhi (v ya) (v yb)));
           line (Isa.const_decl isa (v (tag ^ "u2")) (Printf.sprintf "%s(%s, %s)" unlo (v yc) (v yd)));
           line (Isa.const_decl isa (v (tag ^ "u3")) (Printf.sprintf "%s(%s, %s)" unhi (v yc) (v yd)));
           sline (Isa.storeu_pd isa (st "pl" qoff)
                    (Printf.sprintf "%s(%s, %s, 0x20)" p2f (v (tag ^ "u0")) (v (tag ^ "u2"))));
           sline (Isa.storeu_pd isa (st "pl" (qoff + vw))
                    (Printf.sprintf "%s(%s, %s, 0x20)" p2f (v (tag ^ "u1")) (v (tag ^ "u3"))));
           sline (Isa.storeu_pd isa (st "ph" qoff)
                    (Printf.sprintf "%s(%s, %s, 0x31)" p2f (v (tag ^ "u0")) (v (tag ^ "u2"))));
           sline (Isa.storeu_pd isa (st "ph" (qoff + vw))
                    (Printf.sprintf "%s(%s, %s, 0x31)" p2f (v (tag ^ "u1")) (v (tag ^ "u3"))))
         in
         Buffer.add_string
           buf
           (Printf.sprintf
              "        /* ---- half %s: columns %s, %s -> runs rb[%s], rb[%s] (2 blocks each) ---- */\n"
              (String.uppercase_ascii p) pos_a pos_b pos_a pos_b);
         line
           (Printf.sprintf "double * __restrict__ %s = zout + %d*rb[%s];" (v "pl") (2 * radix) pos_a);
         line
           (Printf.sprintf "double * __restrict__ %s = zout + %d*rb[%s];" (v "ph") (2 * radix) pos_b);
         for l = 0 to 7 do
           line (Isa.const_decl isa (v (string_of_int l)) (Isa.loadu_pd isa (leg_addr "zin" l "Ls" plus)))
         done;
         (* b_m = a_m + a_{m+4}; c_m = (a_m - a_{m+4}) W8^m *)
         for m = 0 to 3 do
           line (Isa.const_decl isa (v ("b" ^ string_of_int m))
                   (Isa.add_pd isa (v (string_of_int m)) (v (string_of_int (m + 4)))));
           line (Isa.const_decl isa (v ("d" ^ string_of_int m))
                   (Isa.sub_pd isa (v (string_of_int m)) (v (string_of_int (m + 4)))))
         done;
         line (Isa.const_decl isa (v "c0") (v "d0"));
         line (Isa.const_decl isa (v "d1i") (Isa.xor_mask_pd isa (Isa.cflip_pd isa (v "d1")) s0t_mask));
         line (Isa.const_decl isa (v "c1") (Isa.mul_pd isa (Isa.add_pd isa (v "d1") (v "d1i")) "_zs0t_rh"));
         line (Isa.const_decl isa (v "c2") (Isa.xor_mask_pd isa (Isa.cflip_pd isa (v "d2")) s0t_mask));
         line (Isa.const_decl isa (v "d3i") (Isa.xor_mask_pd isa (Isa.cflip_pd isa (v "d3")) s0t_mask));
         line (Isa.const_decl isa (v "c3") (Isa.mul_pd isa (Isa.sub_pd isa (v "d3i") (v "d3")) "_zs0t_rh"));
         (* even digits Y0,Y2,Y4,Y6 = DFT4(b); odd digits Y1,Y3,Y5,Y7 = DFT4(c) *)
         r4 ("b0", "b1", "b2", "b3") ("E0", "E1", "E2", "E3");
         r4 ("c0", "c1", "c2", "c3") ("O0", "O1", "O2", "O3");
         (* digit d = 2e (E_e) / 2e+1 (O_e): quartet 0 = Y0..Y3 = E0,O0,E1,O1;
            quartet 1 = Y4..Y7 = E2,O2,E3,O3 *)
         turn ("E0", "O0", "E1", "O1") 0 "q0";
         turn ("E2", "O2", "E3", "O3") 8 "q1")
      [ "a", 0, "k", "k+1"; "b", vw, "k+2", "k+3" ];
    Buffer.add_string buf "    }\n"
  in
  if k.base = "t0tp"
  then (
    (* -- t0tp: the turn lattice as an always_inline BODY (the fused driver
          TU inlines it), stores through rb[]; the exported wrapper hands rb
          over from the tw_im slot (CONTRACT.md 7.1). -- *)
    let body_name = ztt_body_name ~base:"t0tp" ~radix ~bwd:k.bwd in
    body_start := Buffer.length buf;
    Buffer.add_string
      buf
      (Printf.sprintf
         "static __attribute__((always_inline)) inline void %s(\n\
         \    const double * __restrict__ zin, double * __restrict__ zout,\n\
         \    const size_t * __restrict__ rb, size_t Ls, size_t count)\n\
          {\n"
         body_name);
    if radix = 8 then emit_t0tp8_body () else emit_t0tp4_body ();
    Buffer.add_string buf "}\n\n";
    if not body_only
    then (
      (* M4 phase 3: the FROZEN z ABI from Abi.z11_signature; the DERIVED
         silencer list (plain_voids) stays this family's own. *)
      Buffer.add_string
        buf
        (Abi.z11_signature
           ~alias_tolerant:false
           ~symbol:fname
           ~target_attr:isa.Isa.target_attr
           ());
      Buffer.add_string buf (Printf.sprintf "    %s\n" plain_voids);
      Buffer.add_string
        buf
        (Printf.sprintf "    %s(zin, zout, (const size_t *)tw_im, Ls, count);\n}\n" body_name)))
  else (
    (* ── group-looped kinds: static always_inline body + thin group-loop
           wrapper. The body carries NO target attribute (always_inline
           requires the callee's target ⊆ caller's; it inlines into the
           attributed wrapper). ── *)
    let ztt = List.mem k.base [ "tmg"; "tlf"; "tlfi"; "t0d"; "tmgd"; "tld" ] in
    let body_name =
      if ztt
      then ztt_body_name ~base:k.base ~radix ~bwd:k.bwd
      else Printf.sprintf "_zsg%d%s_body" radix (if k.bwd then "b" else "f")
    in
    body_start := Buffer.length buf;
    Buffer.add_string
      buf
      (Printf.sprintf
         "static __attribute__((always_inline)) inline void %s(\n\
         \    const double *%szin, double *%szout,\n\
         \    const double *tw_re, size_t Ls, %ssize_t count)\n\
          {\n"
         body_name
         (* 🔴 The ZTURN-T stages run IN PLACE: the executor and the fused
            drivers call them f(plane, 0, plane, 0, ...) -- the SAME pointer
            for input and output -- so they must not promise the compiler
            their planes are disjoint. The helper is ALWAYS_INLINE, so its
            qualifiers apply to the inlined code regardless of what the
            wrapper says: BOTH levels drop __restrict__. (Measured 2026-08-22
            on the in-place mids: fewer insns AND fewer spills without the
            qualifier — __restrict__ licensed hoisting every load above the
            stores.) *)
         (if ztt then " " else " __restrict__ ")
         (if ztt then " " else " __restrict__ ")
         (if uses "OLs" then "size_t OLs, " else ""));
    (* a twiddle-free group-looped body (tld/tldb) keeps the shared body
       signature; silence its unused stream pointer *)
    if not k.twiddled then Buffer.add_string buf "    (void)tw_re;\n";
    let dag = prepare () in
    if not k.narrow_arms
    then
      emit_col_loop
        ~open_line:
          (Printf.sprintf "    for (size_t k = 0; k + %d <= count; k += %d) {\n" vw vw)
        dag
    else (
      (* il_odd_count_tail.md §3 arms: function-scope k, the wide loop,
         then the SAME DAG at 2 columns (VEX-128) and at 1 column (scalar).
         A count below VW skips the wide loop at its condition (the §3
         low-trip bypass; there is no wide prologue to skip). *)
      Buffer.add_string buf "    size_t k = 0;\n";
      emit_col_loop
        ~open_line:(Printf.sprintf "    for (; k + %d <= count; k += %d) {\n" vw vw)
        dag;
      Buffer.add_string
        buf
        "    /* odd-count arms (il_odd_count_tail.md §3): 2 columns at VEX-128, then 1 \
         column scalar */\n";
      emit_col_loop
        ~nisa:Isa.sse2
        ~open_line:"    for (; k + 2 <= count; k += 2) {\n"
        dag;
      emit_col_loop ~nisa:Isa.scalar ~open_line:"    for (; k < count; ++k) {\n" dag);
    Buffer.add_string buf "}\n\n";
    if not body_only
    then begin
      (* M4 phase 3: the driver wrapper's FROZEN z ABI also comes from
         Abi.z11_signature. *)
      Buffer.add_string
        buf
        (Abi.z11_signature
           ~alias_tolerant:ztt   (* the in-place stages: see the body header *)
           ~symbol:fname
           ~target_attr:isa.Isa.target_attr
           ());
      if k.base = "tlf" || k.base = "tlfi"
      then
        (* distinct in/out pointers: the plane is read at 2*R*Ls per group,
           the packed output written at 2*R*OLs; ONE stream for every group *)
        Buffer.add_string
          buf
          (Printf.sprintf
             "    (void)zin_unused; (void)zout_unused; (void)tw_im; (void)OGs;\n\
             \    const double *ip = zin;\n\
             \    double *op = zout;\n\
             \    for (size_t g = 0; g < Gs; g++) {\n\
             \        %s(ip, op, tw_re, Ls, OLs, count);\n\
             \        ip += 2 * (size_t)%d * Ls;\n\
             \        op += 2 * (size_t)%d * OLs;\n\
             \    }\n\
              }\n"
             body_name
             radix
             radix)
      else if k.base = "t0d" || k.base = "tld"
      then
        (* the PLAIN boundary kinds: distinct in/out pointers (t0d reads the
           caller's z and writes the block-split working buffer; tld/tldb are in
           place when the caller passes one pointer). Gs groups; the group pitch
           is 2*R*Ls for t0d (its columns stride Ls, Gs = 1 at stage 0) and
           2*R*count for tld (a "column" is one R-run, count of them per group). *)
        Buffer.add_string
          buf
          (Printf.sprintf
             "    (void)zin_unused; (void)zout_unused; (void)tw_im; (void)OLs; (void)OGs;\n\
             \    const double *ip = zin;\n\
             \    double *op = zout;\n\
             \    for (size_t g = 0; g < Gs; g++) {\n\
             \        %s(ip, op, tw_re, Ls, count);\n\
             \        ip += 2 * (size_t)%d * %s;\n\
             \        op += 2 * (size_t)%d * %s;\n\
             \    }\n\
              }\n"
             body_name
             radix
             (if k.base = "t0d" then "Ls" else "count")
             radix
             (if k.base = "t0d" then "Ls" else "count"))
      else
        (* the in-place mids (tmg/tmgd, msz): in place on zout (zin voided),
           bp += 2·R·Ls per group; the twiddle cursor bumps per group unless
           the kind's ONE stream serves every group *)
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
             %s\
             \    }\n\
              }\n"
             body_name
             radix
             (if k.tw_group_reset
              then "        /* ZTURN-T: the cursor RESETS per group (CONTRACT.md 3) */\n"
              else Printf.sprintf "        twg += %d;\n" ((radix - 1) * 2 * vw)))
    end);
  if body_only
  then Buffer.sub buf !body_start (Buffer.length buf - !body_start)
  else Buffer.contents buf
;;
