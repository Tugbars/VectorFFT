(* codelet_cil.ml — pipeline-hosted INTERLEAVED-COMPLEX (full-IL) codelets.
 *
 * Full IL is THE mainstream layout, not an alternative: MKL and FFTW both
 * default to interleaved, it is the natural order + the natural K=1 handling
 * standard, it pays no R2C/C2R packing tax, and roughly 80% of users are on
 * it. Split-complex is the specialization for high-throughput batched work.
 * The goal is IL as COMPLETE as FFTW's/MKL's. This module generates the
 * interleaved family the way the rest of the generator works, replacing
 * codelet_zil.ml's hand-scheduled raw-string emission.
 *
 * KINDS: n1 (solo monolithic), n1t (bailey2 stage-1 leaf, four-step
 * transpose fused into the stores), t2 (bailey2 stage-2 mid, streamed VTW2
 * twiddles) — each in BOTH directions.
 *
 * WHAT IS SHARED WITH THE REST OF THE PIPELINE
 * --------------------------------------------
 *   Schedule.Make  — the SR (Starve-Retire) list scheduler, instantiated on
 *                    THIS module's complex IR via the SCHED_NODE signature.
 *                    Replaces codelet_zil's hand A/B interleaving; the same
 *                    scheduler that drives every split-real codelet. MEASURED
 *                    WIN at large radix: -39% reg-reg moves at r32, -49% at
 *                    r64, -34% stack traffic at r16 vs the hand order.
 *   Isa.*          — every intrinsic is built through the ISA layer, so the
 *                    emitted code is width-parametric (AVX2 today, AVX-512
 *                    by passing a different Isa.t) instead of 486 literal
 *                    _mm256_ calls.
 *
 * WHAT IS *NOT* SHARED — Ir / Algsimp / Dft, and the honest reasons
 * -----------------------------------------------------------------
 * Those are REAL-VALUED by construction: the DAG has already split complex
 * into re/im subtrees, and `of_expr` hard-matches the split-cmul shape, so a
 * packed-complex DAG cannot flow through them. Merging packed-complex kinds
 * into Ir.node_kind was measured at ~150-180 exhaustive-match arms across 7
 * modules (dune's dev profile makes non-exhaustive matches fatal).
 *
 * Do NOT read that as "algsimp is unnecessary" — it is NOT (see
 * regalloc.ml §1/§4 and the measurements in zil_pipeline_port.md §11):
 * fma_lift's real job is to leave gcc no mul+add to fuse, so gcc stops
 * re-scheduling off the SU+GH order and stops churning the register
 * allocator. Disabling it costs +8..22% ops on pow2 and +33..71% on
 * odd/prime.
 *
 * This module does not NEED those passes for two specific reasons:
 *   1. pow2 IL is FMA-shaped BY CONSTRUCTION — the twiddle classes fold
 *      sqrt(1/2) into FMAs and BYTW2 is exactly 1 mul + 1 fma (optimal for a
 *      complex multiply). Verified: mul/fma counts identical to the hand
 *      kernels, reg-reg moves stuck at 2-7. There is no FMA-lift headroom
 *      left to recover.
 *   2. For odd radices the same discipline is delivered by the
 *      CONJUGATE-PAIR construction below (dft_cx_odd + cscale_chain: sign
 *      rides in the opcode, magnitude becomes the FMA constant, 0/±1
 *      weights collapse) — see the ODD/PRIME section. What does NOT exist,
 *      on either the IL or the real side, is true Winograd/Rader
 *      multiplication-minimal structure INSIDE a codelet radix; algsimp
 *      never discovered it for the real side either (dft_recurse
 *      hand-builds the same conjugate-pair shape). That remains the open
 *      math-layer question. (An earlier version of this note claimed emit
 *      hard-fails on non-pow2 — stale since the odd-IL arc landed.)
 *
 * Acknowledged duplication: `dft_cx` below re-expresses the DIT recursion
 * that dft.ml already has for the real-valued side. ~40 lines, no cleverness,
 * but it IS a second copy — keep the twiddle-class selection in sync if the
 * real-side one ever changes. CSE is not lost: hash-consing here gives the
 * same sharing Ir gets (verified by bit-identical output vs the hand kernels).
 *
 * IL PRIMITIVES (the only ops with no real-lane equivalent)
 * ---------------------------------------------------------
 * A vector holds vec_width/2 complex as [re,im,re,im,...]:
 *   complex add/sub  = plain vector add/sub          (Isa.add_pd / sub_pd)
 *   c*x + e (real c) = plain FMA                     (Isa.fmadd_pd)
 *   x * (-i)         = xor(cflip x, _M_IM)   fwd quarter-turn
 *   x * (+i)         = xor(cflip x, _M_RE)   bwd quarter-turn
 *   x * (c + i·s)    = fmadd(cvec, x, mul(svec, cflip x))   -- BYTW2 shape,
 *                      cvec/svec either emit-time VLIT constants (n1) or
 *                      loaded from the streamed VTW2 table (t2).
 * ------------------------------------------------------------------
 * MODULE CARD (codelet_cil.ml — grep "MODULE CARD" for the full set)
 * ROLE: interleaved-complex (full-IL) codelet emitter — solo + bailey2,
 * fwd + bwd, pow2 radices.
 * PIPELINE: cx math -> hash-cons -> Schedule.Make(Node).su_schedule ->
 * Isa-parametric emission.
 * PUBLIC SURFACE: emit ~kind ~dir ~radix ~isa ~uarch
 * (gen_main --cil-n1 / --cil-n1t / --cil-t2 [--cil-bwd]).
 * DEPS: Schedule (functor), Isa, Uarch, Expr (elem_ref labels only).
 * GOTCHA 1: `reset ()` MUST run before each codelet — the hash-cons table
 * and tag counter are module-global, exactly like Algsimp.reset.
 * GOTCHA 2: bwd conjugation for t2 is TABLE-SIDE (caller passes a conjugated
 * table); the kernel's BYTW2 is bit-for-bit the forward one, only its
 * POSITION moves (pre-butterfly fwd, post-butterfly bwd).
 * ------------------------------------------------------------------ *)

(* Phase 0 decomposition (2026-08-09, byte-identity gated): the IR, the
 * scheduler instance, the math builders, and the renderer now live in
 * cx_ir.ml / cx_sched.ml / cx_math.ml / cx_render.ml. This module keeps the
 * public surface (kinds, emit, emit_k1) and the emission orchestration.
 * The seam for optimizer passes is between the DAG build (Cx_math.dft_small / dft_chain)
 * and Sched.su_schedule below — a cx pass slots there (cx_passes.ml). *)
open Cx_ir
open Cx_math
open Cx_render
module Sched = Cx_sched.Sched

(* Scheduler dispatch: default = the shared SR (byte-identity of every
   shipped emission preserved); VFFT_CX_SCHED=cpl selects the critical-path
   list scheduler (cx_cpl.ml) — the ILP objective for spill-free tangent
   bodies. Racing decides per slot; wisdom ships the winner. *)
let cx_schedule (uarch : Uarch.t) (assigns : (Expr.elem_ref * Cx_ir.t) list) =
  match Sys.getenv_opt "VFFT_CX_SCHED" with
  | Some "cpl" | Some "cpl2" -> Cx_cpl.schedule uarch assigns
  | Some "asis" -> Cx_cpl.schedule_asis assigns
  | _ -> Sched.su_schedule uarch assigns
;;

(* ═══════════════════════════════════════════════════════════════
 *  KINDS
 *
 *   N1   — solo / monolithic leaf: leg-major z in, leg-major z out.
 *   N1T  — bailey2 stage-1 leaf: same butterfly, but the four-step's
 *          TRANSPOSE is fused into the stores (corner-turn), so stage 2
 *          reads columns contiguously and no separate transpose pass is
 *          needed. Output element (leg p, column k) lands at
 *          zout[2*(k*OLs + p)].
 *   T2   — bailey2 stage-2 mid: legs 1..R-1 pre-twiddled from the
 *          streamed VTW2 table (cursor advances one record-set per
 *          column-group), then the same butterfly. Leg-major stores.
 * ═══════════════════════════════════════════════════════════════ *)
type kind =
  | N1
  | N1T
  | T2

let kind_of_string = function
  | "n1" -> N1
  | "n1t" -> N1T
  | "t2" -> T2
  | s -> failwith (Printf.sprintf "codelet_cil: unknown kind %s (n1 | n1t | t2)" s)
;;

let kind_name = function
  | N1 -> "n1"
  | N1T -> "n1t"
  | T2 -> "t2"
;;

(* DIRECTION.
   n1 / n1t bwd are just the inverse butterfly (unnormalized: no 1/N).
   t2 bwd is the exact inverse of t2 fwd. Since fwd computes
     y = DFT(w (.) x)
   the inverse is
     x = conj(w) (.) IDFT(y)
   i.e. the twiddle moves to AFTER the butterfly and is conjugated. We do NOT
   conjugate in-kernel: the caller passes a conjugated table (the same
   table-side convention the split cascade uses for twspb/twqb), so the BYTW2
   apply is bit-for-bit the forward one and only its POSITION changes. *)
type dir =
  | Fwd
  | Bwd

(* Emit a solo (monolithic, twiddle-free) interleaved n1 codelet.
   ABI: the frozen 11-arg z ABI shared with codelet_zil, so emitted files
   are drop-in against the same benches/drivers. *)
let emit ~(log3 : bool) ~(pretw : bool) ~(turnst : bool) ~(turnst_gs : bool)
      ~(kind : kind)
      ~(dir : dir) ~(blocked : bool) ~(split : (int * int) option)
      ~(radix : int) ~(isa : Isa.t) ~(uarch : Uarch.t)
  : string
  =
  tw_pre := pretw;
  st_turn := turnst || turnst_gs;
  st_turn_gs := turnst_gs;
  (* Required, not optional: an optional arg here cannot be erased (OCaml
     warning 16), and making the policy explicit at every call site is better
     anyway. Only T2 streams a runtime table, so log3 is meaningless on the
     other kinds — refuse rather than silently ignore. *)
  if log3 && kind <> T2
  then
    failwith
      "codelet_cil: --cil-log3 applies to the T2 mid only (it is a sourcing \
       policy for the streamed VTW2 table; n1/n1t carry no runtime twiddles)";
  tw_log3 := log3;
  let vw = isa.Isa.vec_width in
  if vw mod 2 <> 0 then failwith "codelet_cil: interleaved needs an even vec_width";
  let per = vw / 2 in
  (* complex per vector *)
  if (kind = N1T || !st_turn) && per <> 2
  then
    (* The corner-turn store pairs two legs with one permute2f128 (a
       2-complex-per-vector shape). A width-8 vector holds 4 complex and
       needs a 4-way lane shuffle instead — not written yet. *)
    failwith "codelet_cil: n1t corner-turn store is written for 2 complex/vector (avx2)";
  (* RADIX GATE: >= 2 only. Pow2 -> dft_cx, odd -> conjugate pair, and (as
     of 2026-07-29) EVEN COMPOSITES -> dft_small's mixed radix-2 recursion
     bottoming out in the odd builder — the old "would drop legs" refusal
     described dft_cx alone, which the dispatcher no longer exposes to
     mixed halves. *)
  if radix < 2 then failwith "codelet_cil: radix must be >= 2";
  reset ();
  let sign = if dir = Fwd then `Fwd else `Bwd in
  (* Position is independent of direction — see `tw_pre`. `--cil-pretw` forces
     PRE on a backward T2, which is the combination the pure-IL inverse needs. *)
  let pre_tw = kind = T2 && (dir = Fwd || !tw_pre) in
  let post_tw = kind = T2 && dir = Bwd && not !tw_pre in
  (* COMPLETE-IR (2026-08-09): monolithic inputs are CLoad nodes carrying
     their symbolic address — the load edge prints FROM the DAG instead of
     inventing the string. Created in the same order cin was, so every tag
     (= every zN name) is unchanged: byte-identity by construction. *)
  (* WING-T2 FULL KERNEL (VFFT_CX_WING + tangent + T2 + R16 + Fwd): the wing
     construction owns its loads AND the streamed BYTW2 ingest, emitted in the
     ORIGIN's listing order, so asis reproduces the hand kernel's interleaved
     load/ingest/butterfly stream (peak 15). Bypasses the normal input+ingest
     pre-pass, which front-loads all 16 ingests and forces peak 16. *)
  let use_wing_t2 =
    kind = T2 && dir = Fwd && radix = 16 && !Cx_math.tangent && !Cx_math.wing_enabled
  in
  let outs =
    if use_wing_t2
    then Cx_math.dft_cx16_wing_t2 ()
    else (
      let inputs =
        Array.init radix (fun i ->
          (* T2 fwd PRE-twiddles legs 1..R-1 from the streamed table; leg 0 is
             untwiddled (w^0 = 1), which is why records start at leg 1. *)
          if pre_tw && i > 0 then ctwl i (cload (AZinLeg i)) else cload (AZinLeg i))
      in
      dft_small ~sign radix inputs)
  in
  (* T2 bwd POST-twiddles: conj(w) (.) IDFT(y). Same BYTW2 apply, same table
     slots, just after the butterfly — see the `dir` note above. *)
  let outs =
    if post_tw then Array.mapi (fun i e -> if i > 0 then ctwl i e else e) outs else outs
  in
  (* Label outputs with Expr.elem_ref so the shared scheduler can identify
     sinks; only the index is meaningful here (one complex output per leg). *)
  let assigns = Array.to_list (Array.mapi (fun i e -> Expr.Output (i, true), e) outs) in
  (* THE PIPELINE SEAM: every cil body flows through the cx pass cascade
     between construction and scheduling — cil is pipeline-hosted. *)
  let assigns = Cx_pipeline.prepare_codelet ~who:(Printf.sprintf "r%d_%s" radix (kind_name kind)) ~uarch assigns in
  let scheduled = cx_schedule uarch assigns in
  let tbl : consts = Hashtbl.create 16 in
  (* Render the body first: it populates the constant table that the file
     preamble must declare. *)
  let body = Buffer.create 4096 in
  (* ─── ONE SCHEDULED PASS ──────────────────────────────────────────
     Build a sub-DAG, run it through the SHARED scheduler, and emit
     loads / defs / stores for it. Used by the BLOCKED construction,
     which is several passes joined through a spill array.

     Each pass calls `reset ()`, so tag numbering (and therefore the
     `zN` variable names) RESTARTS — every pass must therefore be
     emitted inside its own C brace scope, and fully emitted before the
     next pass begins. Same discipline as sterm2's two DAGs. *)
  (* Per-pass lazy load/store (VFFT_CX_LAZYLOAD / VFFT_CX_LAZYSTORE, default
     OFF => the up-front loads + batched stores below => byte-identical to
     every existing blocked kernel). Ports the mono peak-pressure fix into the
     blocked passes: interleaving each pass's loads with compute and storing
     its outputs the moment they are ready frees registers so gcc hoists the
     loop-invariant constants (the R16 lesson: rip-const 27->4). ~lazy_store
     is opt-in per caller because the corner-turned N1T store pairs groups and
     must NOT be interleaved per-index. *)
  let emit_pass
        ~(lazy_store : bool)
        ~(label : string)
        ~(nin : int)
        ~(laddr_of : int -> caddr)
        ~(build : t array -> t array)
        ~(store : int -> t -> unit)
    : unit
    =
    reset ();
    let ll = Sys.getenv_opt "VFFT_CX_LAZYLOAD" = Some "1" in
    let ls = lazy_store && Sys.getenv_opt "VFFT_CX_LAZYSTORE" = Some "1" in
    (* COMPLETE-IR: pass inputs are CLoad nodes carrying their address
       (same creation order cin had ⇒ same tags ⇒ same zN names). *)
    let ins = Array.init nin (fun i -> cload (laddr_of i)) in
    let outs = build ins in
    let assigns =
      Array.to_list (Array.mapi (fun i e -> Expr.Output (i, true), e) outs)
    in
    let assigns = Cx_pipeline.prepare_codelet ~who:label ~uarch assigns in
    let sch = cx_schedule uarch assigns in
    Buffer.add_string body (Printf.sprintf "        { /* %s */\n" label);
    (* up-front loads only when NOT lazy *)
    if not ll
    then
      Array.iteri
        (fun i (e : t) ->
           Buffer.add_string
             body
             (Printf.sprintf
                "        %s\n"
                (Isa.const_decl
                   isa
                   (Printf.sprintf "z%d" e.tag)
                   (Isa.loadu_pd isa (addr_str (laddr_of i))))))
        ins;
    let load_emitted : (int, unit) Hashtbl.t = Hashtbl.create 64 in
    let emit_load_p (l : t) =
      match l.node with
      | CLoad a when ll && not (Hashtbl.mem load_emitted l.tag) ->
        Hashtbl.replace load_emitted l.tag ();
        Buffer.add_string
          body
          (Printf.sprintf
             "        %s\n"
             (Isa.const_decl isa (Printf.sprintf "z%d" l.tag) (Isa.loadu_pd isa (addr_str a))))
      | _ -> ()
    in
    let stored : (int, unit) Hashtbl.t = Hashtbl.create 32 in
    let seen : (int, unit) Hashtbl.t = Hashtbl.create 256 in
    List.iter
      (fun ((eref : Expr.elem_ref option), (e : t)) ->
         if ll then List.iter emit_load_p (Cx_sched.Node.preds e);
         (match e.node with
          | CIn _ | CLoad _ -> ()
          | _ ->
            if not (Hashtbl.mem seen e.tag)
            then (
              Hashtbl.replace seen e.tag ();
              Buffer.add_string
                body
                (Printf.sprintf
                   "        %s\n"
                   (Isa.const_decl isa (Printf.sprintf "z%d" e.tag) (render isa tbl e)))));
         if ls
         then
           (match eref with
            | Some (Expr.Output (i, _)) -> store i e; Hashtbl.replace stored i ()
            | _ -> ()))
      sch;
    Array.iteri (fun i (e : t) -> if not (Hashtbl.mem stored i) then store i e) outs;
    Buffer.add_string body "        }\n"
  in
  (* ─── BLOCKED (2-pass) construction ──────────────────────────────
     Straight-line radix-R needs R values live at once; there are only 16
     vector registers, so from R=16 up gcc spills hard — MEASURED stack
     traffic per kernel: r8 12-14, r16 35-53, r32 158-197, r64 537-554.
     Blocking splits the DIT recursion at its natural seam: the even-leg
     half-DFT and the odd-leg half-DFT each need only R/2 live, are emitted
     as separate scheduled passes, and park to a small function-scope spill
     array; a third pass reloads pairs and runs the top-level butterfly.
     Peak live drops R -> R/2, which is what makes r32/r64 usable as chain
     stages at all. Mirrors codelet_zil's emit_z_blocked_body, but each pass
     is scheduled by the SHARED scheduler and the twiddles stay compile-time
     constants. *)
  let emit_blocked () =
    (* Cooley-Tukey split R = m * p, decimating legs by residue mod m:
         n = a*m + i    ->   A_i[j] = DFT_p over a of x[a*m+i]
         X[j + p*k2]    =   DFT_m over i of ( A_i[j] * W_R^{i*j} )
       PASS 1 emits m sub-DFTs of size p (each needs only p live) parking to
       S[i*p + j]; PASS 2 emits p groups, each reloading m values, applying
       the compile-time W_R^{i*j}, and running an m-point DFT.
       m = 2 is the plain halving (its DFT_2 is exactly butterfly_pair);
       m = 8 at R=64 is the 8x8 form — needed because halving 64 leaves
       32-point halves that still spill. Peak live goes R -> max(p, m). *)
    (* THE SPLIT IS A PLAN INPUT, NOT AN EMITTER DECISION.
       For pow2 the historical rule is kept verbatim so every existing kernel
       stays byte-identical. For a NON-pow2 radix there is no defensible
       default — `radix / m` with a hand-picked m would either truncate (odd
       radix, integer division) or re-commit the "squarest split" mistake this
       same file records under emit_k1: a factorization invented inside an
       emitter, which contradicted the calibrated chains and caused measured
       losses. So a non-pow2 blocked emission REQUIRES an explicit ~split, and
       the emitter only VALIDATES it. *)
    let m, p =
      match split with
      | Some (sm, sp) ->
        if sm < 2 || sp < 2 || sm * sp <> radix
        then
          failwith
            (Printf.sprintf
               "codelet_cil: --cil-split %d.%d does not factor radix %d (need \
                m,p >= 2 and m*p = radix)"
               sm
               sp
               radix);
        sm, sp
      | None ->
        if radix land (radix - 1) <> 0
        then
          failwith
            (Printf.sprintf
               "codelet_cil: --cil-blocked at radix %d needs an explicit \
                --cil-split m.p. There is no defensible default for a non-pow2 \
                radix, and the factorization is a PLAN input, not an emitter \
                decision."
               radix)
        else (
          let m = if radix >= 64 then 8 else 2 in
          m, radix / m)
    in
    if m * p <> radix then failwith "codelet_cil: blocked split must multiply to radix";
    let pi = 4.0 *. atan 1.0 in
    let sgn = if dir = Fwd then -1.0 else 1.0 in
    (* PASS 1: sub-DFT i over legs { a*m + i } *)
    for i = 0 to m - 1 do
      emit_pass
        ~lazy_store:true
        ~label:
          (Printf.sprintf
             "PASS 1.%d: legs {a*%d+%d} -> S[%d..%d]"
             i m i (i * p) ((i * p) + p - 1))
        ~nin:p
        ~laddr_of:(fun a -> AZinLeg ((a * m) + i))
        ~build:(fun ins ->
          let ins =
            if pre_tw
            then
              Array.mapi
                (fun a x ->
                   let l = (a * m) + i in
                   if l > 0 then ctwl l x else x)
                ins
            else ins
          in
          dft_small ~sign p ins)
        ~store:(fun j e ->
          let ad = AS (vw * ((i * p) + j)) in
          let (_ : t) = cstore ad e in
          Buffer.add_string
            body
            (Printf.sprintf
               "        %s;\n"
               (render_store isa ad (Printf.sprintf "z%d" e.tag))))
    done;
    (* Shared PASS-2 math: twiddle group j by W_R^{i*j}, DFT_m, and (T2 bwd)
       post-twiddle — factored so the plain and TURNED store paths below run
       the IDENTICAL dataflow and differ only in the store edge. *)
    let pass2_math ~(jv : int) (ins : t array) : t array =
      let o =
        if m = 2
        then (
          (* m=2 is a single top-level butterfly, so use the CLASS-aware
             form: it turns W^{R/4} into a free rotation and folds the
             W^{R/8} pair into FMAs, which a general complex multiply
             would not. Keeps the blocked output bit-identical to the
             monolithic one. *)
          let a, b = butterfly_pair ~sign ~n:radix ~k:jv ins.(0) ins.(1) in
          [| a; b |])
        else (
          let tw =
            Array.mapi
              (fun i x ->
                 let e = i * jv mod radix in
                 if e = 0
                 then x
                 else if 4 * e = radix
                 then (if sign = `Fwd then crot x else crotp x)
                 else (
                   let a = sgn *. 2.0 *. pi *. float_of_int e /. float_of_int radix in
                   ctw (cos a) (sin a) x))
              ins
          in
          dft_small ~sign m tw)
      in
      if post_tw
      then
        Array.mapi
          (fun k2 e ->
             let l = jv + (p * k2) in
             if l > 0 then ctwl l e else e)
          o
      else o
    in
    let turned = kind = N1T || !st_turn in
    if turned && !st_turn_gs
    then
      failwith
        "codelet_cil: --cil-blocked does not implement the leg-strided (t2tg) \
         turn; only the contiguous corner-turn is supported blocked.";
    if turned && p mod 2 <> 0
    then
      failwith
        (Printf.sprintf
           "codelet_cil: blocked turned stores pair pass-2 groups (j, j+1), \
            which needs an EVEN p; split %d.%d has p = %d. Pick an even-p \
            split."
           m p p);
    if not turned
    then
      (* PASS 2 (plain leg-major stores): per j, one scheduled group. *)
      for j = 0 to p - 1 do
        emit_pass
          ~lazy_store:true
          ~label:(Printf.sprintf "PASS 2.%d: S[i*%d+%d] -> X[%d + %d*k2]" j p j j p)
          ~nin:m
          ~laddr_of:(fun i -> AS (vw * ((i * p) + j)))
          ~build:(fun ins -> pass2_math ~jv:j ins)
          ~store:(fun k2 e ->
            let ad = AZoutLeg (j + (p * k2)) in
            let (_ : t) = cstore ad e in
            Buffer.add_string
              body
              (Printf.sprintf
                 "        %s;\n"
                 (render_store isa ad (Printf.sprintf "z%d" e.tag))))
      done
    else
      (* ─── PASS 2, TURNED (N1T / t2t): the corner-turn pairs ADJACENT legs
         (l, l+1), but pass 2.j produces legs {j + p*k2} — stride p apart.
         Adjacent legs live in ADJACENT groups: l = j+p*k2 pairs with
         (j+1)+p*k2 from group j+1. So emit pass 2 as PASS-PAIRS (j, j+1):
         one scheduled sub-DAG computes both groups (2m outputs live at the
         pass tail — fits the file for m <= 4: 2m + temps ~ 12 ymm at 4.8;
         an m = 8 split would hold 16 + temps and start spilling, which is
         the caller's split choice, not a guard), then each k2 emits the
         same paired permute2f128 stores the monolithic N1T edge uses:
         [c_k legs l,l+1] at zout[2*(k*OLs + l)] (0x20) and the c_{k+1}
         twin (0x31). j even and p even make every l = j + p*k2 even —
         exactly the monolithic lattice. Group-A names are stashed by the
         store callback until their group-B partner arrives. *)
      for jj = 0 to (p / 2) - 1 do
        let j = 2 * jj in
        (* Group-A NODES stashed until the group-B partner arrives; the
           regroup is then CTurn nodes + CStore at AZoutTurn — same DATA
           form as the monolithic corner-turn edge. *)
        let a_nodes : t option array = Array.make m None in
        emit_pass
          ~lazy_store:false
          ~label:
            (Printf.sprintf
               "PASS 2.%d+%d TURNED: S[i*%d+{%d,%d}] -> columns k,k+1"
               j (j + 1) p j (j + 1))
          ~nin:(2 * m)
          ~laddr_of:(fun idx ->
            let i = idx mod m
            and g = idx / m in
            AS (vw * ((i * p) + j + g)))
          ~build:(fun ins ->
            let ga = Array.sub ins 0 m
            and gb = Array.sub ins m m in
            Array.append (pass2_math ~jv:j ga) (pass2_math ~jv:(j + 1) gb))
          ~store:(fun idx e ->
            if idx < m
            then a_nodes.(idx) <- Some e
            else (
              let k2 = idx - m in
              let l = j + (p * k2) in
              let a =
                match a_nodes.(k2) with
                | Some a -> a
                | None -> failwith "codelet_cil: turned pass-pair stash miss"
              in
              let ta = cturn a e 0x20
              and tb = cturn a e 0x31 in
              let (_ : t) = cstore (AZoutTurn (l, 0)) ta
              and (_ : t) = cstore (AZoutTurn (l, 1)) tb in
              Buffer.add_string
                body
                (Printf.sprintf
                   "        %s;\n        %s;\n"
                   (render_store isa (AZoutTurn (l, 0)) (render isa tbl ta))
                   (render_store isa (AZoutTurn (l, 1)) (render isa tbl tb)))))
      done
  in
  (* The old refusal ("emit_blocked never inspects kind") is RESOLVED for the
     contiguous corner-turn: blocked N1T/t2t emit pass-pairs with the paired
     permute2f128 store edge (2026-08-05, il_coverage_plan.md E9). The
     leg-strided t2tg turn and odd-p splits still refuse loudly inside
     emit_blocked itself. *)
  mono_spill_slots := 0;
  (* lazy-store bookkeeping is shared between the mono emit loop (which stores
     outputs inline) and the store edge (which skips those). Hoisted here so
     both see it. *)
  let lazy_stores =
    (use_wing_t2 || Sys.getenv_opt "VFFT_CX_LAZYSTORE" = Some "1")
    && not blocked && not !st_turn && (kind = T2 || kind = N1)
  in
  let stored_inline : (int, unit) Hashtbl.t = Hashtbl.create 32 in
  if blocked
  then emit_blocked ()
  else (
  (* cx_spill plan (VFFT_CX_SPILL=<budget>, default OFF => None => the plain
     byte-identical loop below). When present, an S[] round-trip caps peak
     register pressure so gcc keeps a free register for constant hoisting.
     MONO path only — blocked already parks halves to its own S[]. *)
  let spill_plan = Cx_spill.plan scheduled in
  mono_spill_slots := (match spill_plan with Some p -> p.Cx_spill.nslots | None -> 0);
  let seen : (int, unit) Hashtbl.t = Hashtbl.create 256 in
  (* tag -> current C name; reloads install a fresh name that later uses pick up *)
  let names : (int, string) Hashtbl.t = Hashtbl.create 256 in
  let cur_name t = match Hashtbl.find_opt names t with Some s -> s | None -> Printf.sprintf "z%d" t in
  let reload_ctr = ref 0 in
  let sarr slot = Printf.sprintf "S[%d]" (vw * slot) in
  (* LAZY LOAD MATERIALISATION (VFFT_CX_LAZYLOAD=1, default OFF): the input
     leg loads are normally emitted ALL up front (the load-edge loop below),
     which pins radix vectors live from the first instruction — the mono
     tangent bodies peak at radix live and lose gcc's constant-hoist register.
     With lazy loads on, each CLoad is emitted just before its first consumer
     in scheduled order, matching the hand kernel's interleaved load/compute
     (peak radix -> ~radix-1). OFF keeps every existing kernel byte-identical:
     the up-front loop still runs and this set stays empty. *)
  let lazy_loads = Sys.getenv_opt "VFFT_CX_LAZYLOAD" = Some "1" && not blocked in
  let load_emitted : (int, unit) Hashtbl.t = Hashtbl.create 64 in
  let emit_load (l : t) =
    match l.node with
    | CLoad a when lazy_loads && not (Hashtbl.mem load_emitted l.tag) ->
      Hashtbl.replace load_emitted l.tag ();
      Buffer.add_string
        body
        (Printf.sprintf
           "        %s\n"
           (Isa.const_decl isa (Printf.sprintf "z%d" l.tag) (Isa.loadu_pd isa (addr_str a))))
    | _ -> ()
  in
  (* LAZY STORES (auto-on for the wing-T2 full kernel; also VFFT_CX_LAZYSTORE=1):
     emit each output's store the moment its value is defined, instead of
     batching all 16 stores after the body. Batched stores keep every output
     live to the end, which forces peak pressure and evicts gcc's hoisted
     loop-invariant constants (measured: rip-const 19 batched -> 4 interleaved,
     matching the hand kernel). Leg-major T2/N1 store form only (the wing is a
     T2 fwd); turned/blocked keep the batched edge. Bindings hoisted above. *)
  List.iteri
    (fun pos ((eref : Expr.elem_ref option), (e : t)) ->
       (* lazy loads: materialise any not-yet-emitted CLoad this node reads *)
       if lazy_loads then List.iter emit_load (Cx_sched.Node.preds e);
       (* reloads scheduled BEFORE this position: pull each evicted value back
          into a fresh SSA name and repoint its tag *)
       (match spill_plan with
        | Some pl ->
          (match Hashtbl.find_opt pl.Cx_spill.reload_before pos with
           | Some rs ->
             List.iter
               (fun (t, slot) ->
                  let nm = Printf.sprintf "z%d_r%d" t (let c = !reload_ctr in incr reload_ctr; c) in
                  Buffer.add_string
                    body
                    (Printf.sprintf
                       "        %s\n"
                       (Isa.const_decl isa nm (Isa.loadu_pd isa (sarr slot))));
                  Hashtbl.replace names t nm)
               rs
           | None -> ())
        | None -> ());
       (match e.node with
        | CIn _ | CLoad _ -> () (* materialized by the load edge *)
        | _ ->
          if not (Hashtbl.mem seen e.tag)
          then (
            Hashtbl.replace seen e.tag ();
            Buffer.add_string
              body
              (Printf.sprintf
                 "        %s\n"
                 (Isa.const_decl isa (Printf.sprintf "z%d" e.tag) (render ~name:cur_name isa tbl e)))));
       (* spills scheduled AFTER this position: store the value to its slot *)
       (match spill_plan with
        | Some pl ->
          (match Hashtbl.find_opt pl.Cx_spill.spill_after pos with
           | Some ss ->
             List.iter
               (fun (t, slot) ->
                  Buffer.add_string
                    body
                    (Printf.sprintf "        %s;\n" (Isa.storeu_pd isa (sarr slot) (cur_name t))))
               ss
           | None -> ())
        | None -> ());
       (* lazy store: if this node is an output sink, store it now and free it *)
       if lazy_stores
       then
         (match eref with
          | Some (Expr.Output (i, _)) ->
            let (_ : t) = cstore (AZoutLeg i) e in
            Buffer.add_string
              body
              (Printf.sprintf
                 "        %s;\n"
                 (render_store isa (AZoutLeg i) (cur_name e.tag)));
            Hashtbl.replace stored_inline i ()
          | _ -> ()))
    scheduled);
  (* ─── ODD-COUNT TAIL body (docs/roadmap/tail_handling/il_odd_count_tail.md
     §3): the SAME scheduled DAG re-rendered at Isa.sse2 — one complex per
     iteration, emitted INLINE in the enclosing avx2,fma function (VEX-128,
     no AVX↔SSE transition), no scratch, no call, no duplicated column.
     Rendered BEFORE the preamble is assembled so its 1-lane constants land
     in `tbl` (declared per-entry as __m128d). MONOLITHIC kernels only —
     blocked keeps its even-count contract (doc §4d, open question).
     At per = 2 the leftover is exactly ONE column whose index is EVEN, so
     the T2 cursor's (k / per) group arithmetic and record lane 0 hold
     unchanged; the VTW2 record is already narrow-readable ([c,c] at off,
     [-s,+s] at off + tw_vw) — only the ADDRESSING must keep the wide
     geometry, hence ~tw_vw below. Own C block + own constants = the RA
     mitigation emit_c.ml §4052/4058 uses (hot loop must stay unchanged). *)
  let body_n = Buffer.create 2048 in
  if not blocked
  then (
    let nisa = Isa.sse2 in
    if kind = T2
    then
      Buffer.add_string
        body_n
        (Printf.sprintf
           "        const double *twp = tw_re + (k / %d) * (size_t)%d;\n"
           per
           ((radix - 1) * 2 * vw));
    if kind = T2 && !tw_log3
    then emit_log3_prologue ~tw_vw:vw ~msuf:"_n" body_n nisa radix;
    for l = 0 to radix - 1 do
      Buffer.add_string
        body_n
        (Printf.sprintf
           "        %s\n"
           (Isa.const_decl
              nisa
              (Printf.sprintf "z%d" (cload (AZinLeg l)).tag)
              (Isa.loadu_pd nisa (addr_str (AZinLeg l)))))
    done;
    let seen_n : (int, unit) Hashtbl.t = Hashtbl.create 256 in
    List.iter
      (fun ((_ : Expr.elem_ref option), (e : t)) ->
         match e.node with
         | CIn _ | CLoad _ -> ()
         | _ ->
           if not (Hashtbl.mem seen_n e.tag)
           then (
             Hashtbl.replace seen_n e.tag ();
             Buffer.add_string
               body_n
               (Printf.sprintf
                  "        %s\n"
                  (Isa.const_decl
                     nisa
                     (Printf.sprintf "z%d" e.tag)
                     (render ~tw_vw:vw ~msuf:"_n" nisa tbl e)))))
      scheduled;
    match (if !st_turn then N1T else kind) with
    | N1 | T2 ->
      (* the wide edge below creates the CStore node; the tail prints the
         same address form at narrow width *)
      Array.iteri
        (fun l (e : t) ->
           Buffer.add_string
             body_n
             (Printf.sprintf
                "        %s;\n"
                (render_store nisa (AZoutLeg l) (Printf.sprintf "z%d" e.tag))))
        outs
    | N1T ->
      (* one complex per leg: the corner-turn (and the t2tg leg stride) is
         pure addressing at this width — no permutes, no pairing. The wide
         edge below owns the CStore nodes; the tail prints the col-0 form. *)
      Array.iteri
        (fun l (e : t) ->
           let a = if !st_turn_gs then AZoutTurnG (l, 0) else AZoutTurn (l, 0) in
           Buffer.add_string
             body_n
             (Printf.sprintf
                "        %s;\n"
                (render_store nisa a (Printf.sprintf "z%d" e.tag))))
        outs);
  let buf = Buffer.create 8192 in
  Buffer.add_string
    buf
    (Printf.sprintf
       "/* Auto-generated by vfft_v2 — INTERLEAVED-COMPLEX (full-IL) family,\n\
       \ * PIPELINE-HOSTED (codelet_cil.ml; docs/roadmap/zil_pipeline_port.md §11).\n\
       \ * radix-%d %s: %d complex per %d-bit vector. Body scheduled by the\n\
       \ * SHARED SR scheduler (Schedule.Make over the complex IR) and rendered\n\
       \ * through the ISA layer, so the same source emits AVX2 / AVX-512.\n\
       \ * %s\n\
       \ * %s */\n"
       radix
       (match kind with
        | N1 -> "solo n1 (natural order in/out, twiddle-free)"
        | N1T -> "bailey2 stage-1 leaf n1t (four-step TRANSPOSE fused into the stores)"
        | T2 -> "bailey2 stage-2 mid t2 (streamed VTW2 twiddles, BYTW2 apply)")
       per
       (vw * 64)
       (match kind with
        | N1 -> "tw_re/tw_im unused."
        | N1T ->
          "Stores are corner-turned: output (leg p, column k) -> zout[2*(k*OLs + p)],\n\
          \ * so stage 2 reads whole columns contiguously and no separate transpose\n\
          \ * pass is needed. tw_re/tw_im unused."
        | T2 ->
          Printf.sprintf
            "tw_re = VTW2 records, cos-first and sign-folded: per column-group, per\n\
            \ * leg 1..R-1, one %d-double record [c x%d][-s,+s ...]. Cursor advances\n\
            \ * %d doubles per group; BYTW2 = fmadd(c, x, mul(s, cflip x)) — ONE\n\
            \ * data-side shuffle, zero table-side work. tw_im unused."
            (2 * vw)
            vw
            ((radix - 1) * 2 * vw))
       (if blocked
        then Printf.sprintf "CONTRACT: count %% %d == 0 (%d columns per iteration)." per per
        else
          Printf.sprintf
            "count: ANY >= 1 — %d columns per wide iteration, inline VEX-128\n\
            \ * odd-count tail for the leftover (il_odd_count_tail.md §3)."
            per));
  Buffer.add_string buf "#include <immintrin.h>\n#include <stddef.h>\n\n";
  (* Only the quarter-turn mask this direction actually uses — emitting both
     would leave an unused static const (warning noise). The monolithic tail
     re-renders the DAG at Isa.sse2 and needs the __m128d twin. *)
  if dir = Fwd
  then (
    Buffer.add_string buf (Isa.im_mask_decl isa "_M_IM");
    Buffer.add_string buf "  /* negate im lanes: x*(-i) */\n";
    if not blocked
    then (
      Buffer.add_string buf (Isa.im_mask_decl Isa.sse2 "_M_IM_n");
      Buffer.add_string buf "  /* tail twin */\n"))
  else (
    Buffer.add_string buf (Isa.re_mask_decl isa "_M_RE");
    Buffer.add_string buf "  /* negate re lanes: x*(+i) */\n";
    if not blocked
    then (
      Buffer.add_string buf (Isa.re_mask_decl Isa.sse2 "_M_RE_n");
      Buffer.add_string buf "  /* tail twin */\n"));
  Buffer.add_string buf (emit_const_decls isa tbl);
  Buffer.add_string
    buf
    (Printf.sprintf
       "\n\
        __attribute__((target(\"%s\")))\n\
        void radix%d_z_%s_%s_%s(\n\
       \    const double * __restrict__ zin,\n\
       \    const double * __restrict__ zin_unused,\n\
       \    double       * __restrict__ zout,\n\
       \    double       * __restrict__ zout_unused,\n\
       \    const double * tw_re, const double * tw_im,\n\
       \    size_t Ls, size_t Gs, size_t OLs, size_t OGs, size_t count)\n\
        {\n\
       \    (void)zin_unused; (void)zout_unused; (void)tw_im; (void)Gs;%s%s\n"
       isa.Isa.target_attr
       radix
       (kind_name kind
        ^ (if blocked then "b" else "")
        ^ (if !tw_pre && dir = Bwd then "p" else "")
        ^ (if !st_turn then "t" else "")
        ^ (if !st_turn_gs then "g" else "")
        ^ if !tw_log3 then "_log3" else "")
       (if dir = Fwd then "fwd" else "bwd")
       isa.Isa.name
       (if !st_turn_gs then "" else " (void)OGs;")
       (if kind = T2 then "" else " (void)tw_re;"));
  if blocked
  then
    Buffer.add_string
      buf
      (Printf.sprintf
         "    double S[%d];  /* half-DFT spill: function-scope, L1-hot across \
          iterations */\n"
         (vw * radix))
  else if !mono_spill_slots > 0
  then
    Buffer.add_string
      buf
      (Printf.sprintf
         "    double S[%d];  /* cx_spill Belady scratch (peak-pressure cap) */\n"
         (vw * !mono_spill_slots));
  Buffer.add_string
    buf
    (if blocked
     then Printf.sprintf "    for (size_t k = 0; k + %d <= count; k += %d) {\n" per per
     else
       (* k hoisted so the tail loop below resumes it; blocked keeps the old
          form (no tail there) and stays byte-identical. *)
       Printf.sprintf "    size_t k = 0;\n    for (; k + %d <= count; k += %d) {\n" per per);
  (* T2's streamed cursor: one record-set per column-group. *)
  if kind = T2
  then
    Buffer.add_string
      buf
      (Printf.sprintf
         "        const double *twp = tw_re + (k / %d) * (size_t)%d;\n"
         per
         ((radix - 1) * 2 * vw));
  (* LOG3 binds every leg's record up front — loaded for power-of-two legs,
     derived for the rest — so the butterflies below reference names instead
     of the table. Loop-invariant per column-group, hence off the data
     critical path. *)
  if kind = T2 && !tw_log3 then emit_log3_prologue buf isa radix;
  (* Blocked carries its own per-pass loads and stores; the monolithic form
     needs the leg load edge here — UNLESS lazy-load materialisation is on
     (VFFT_CX_LAZYLOAD=1), in which case each load is emitted just before its
     first consumer in the scheduled body (peak-pressure cap; see the mono
     emit loop). Default OFF => this loop runs => byte-identical. *)
  if not blocked && not (Sys.getenv_opt "VFFT_CX_LAZYLOAD" = Some "1")
  then
    for l = 0 to radix - 1 do
      Buffer.add_string
        buf
        (Printf.sprintf
           "        %s\n"
           (Isa.const_decl
              isa
              (Printf.sprintf "z%d" (cload (AZinLeg l)).tag)
              (Isa.loadu_pd isa (addr_str (AZinLeg l)))))
    done;
  Buffer.add_buffer buf body;
  (* Store edge (blocked emits its own inside PASS 2). Dispatch on the store
     FORM, not the kind: `--cil-turnst` gives a T2 the corner-turned store. *)
  if not blocked then
  (match (if !st_turn then N1T else kind) with
   | N1 | T2 ->
     (* leg-major: leg l's `per` columns stay contiguous. COMPLETE-IR: the
        store is a CStore NODE (address in the DAG); built post-schedule so
        no existing tag shifts, printed via render_store (addr_str carries
        the byte-identity contract). *)
     Array.iteri
       (fun l (e : t) ->
          if not (Hashtbl.mem stored_inline l)
          then (
            let (_ : t) = cstore (AZoutLeg l) e in
            Buffer.add_string
              buf
              (Printf.sprintf
                 "        %s;\n"
                 (render_store isa (AZoutLeg l) (Printf.sprintf "z%d" e.tag)))))
       outs
   | N1T ->
     if !st_turn_gs
     then
       (* LEG-STRIDED turn (T2TG): legs sit at stride OGs, so they are NOT
          contiguous and the paired full-width store below would interleave
          the wrong legs. Every leg's two columns scatter as two 128-bit
          halves instead (the odd-tail pattern applied to all legs):
          (leg p, col k) -> zout[2*(k*OLs + p*OGs)]. Costs 2R narrow stores
          vs R wide — the price of the chain-bwd middle stage; measured at
          the plan level, not assumed away. *)
       (* COMPLETE-IR: the halves are CLo/CHi nodes, the scatters CStore
          nodes at AZoutTurnG — rendered narrow, byte-identical strings. *)
       Array.iteri
         (fun l (e : t) ->
            let lo = clo e
            and hi = chi e in
            let (_ : t) = cstore (AZoutTurnG (l, 0)) lo
            and (_ : t) = cstore (AZoutTurnG (l, 1)) hi in
            Buffer.add_string
              buf
              (Printf.sprintf
                 "        %s;\n        %s;\n"
                 (render_store Isa.sse2 (AZoutTurnG (l, 0)) (render Isa.sse2 tbl lo))
                 (render_store Isa.sse2 (AZoutTurnG (l, 1)) (render Isa.sse2 tbl hi))))
         outs
     else (
     (* CORNER-TURN (the four-step transpose, fused into the stores).
        Each output vector holds one leg's 2 columns: out_p = [c_k, c_{k+1}].
        Pairing legs p,p+1 and swapping 128-bit lanes regroups them into
        [leg p, leg p+1] of ONE column — so column k's legs land
        contiguously at zout[2*(k*OLs + p)]. Two stores per leg-pair, both
        full-width: no scalar tail, no separate transpose pass. *)
     let n = Array.length outs in
     let l = ref 0 in
     (* pairs of legs: one permute2f128 per store, both full width.
        COMPLETE-IR: the lane regroups are CTurn nodes, the paired writes
        CStore nodes at AZoutTurn — the four-step transpose is DATA now. *)
     while !l + 1 < n do
       let ta = cturn outs.(!l) outs.(!l + 1) 0x20
       and tb = cturn outs.(!l) outs.(!l + 1) 0x31 in
       let (_ : t) = cstore (AZoutTurn (!l, 0)) ta
       and (_ : t) = cstore (AZoutTurn (!l, 1)) tb in
       Buffer.add_string
         buf
         (Printf.sprintf
            "        %s;\n        %s;\n"
            (render_store isa (AZoutTurn (!l, 0)) (render isa tbl ta))
            (render_store isa (AZoutTurn (!l, 1)) (render isa tbl tb)));
       l := !l + 2
     done;
     (* ODD RADIX: the last leg has no partner to swap lanes with, so its two
        columns are scattered as two 128-bit stores instead of one paired
        permute2f128. N1T already refuses anything but 2 complex/vector
        (checked above), so a 128-bit half IS exactly one column. *)
     if !l < n
     then (
       let lo = clo outs.(!l)
       and hi = chi outs.(!l) in
       let (_ : t) = cstore (AZoutTurn (!l, 0)) lo
       and (_ : t) = cstore (AZoutTurn (!l, 1)) hi in
       Buffer.add_string
         buf
         (Printf.sprintf
            "        %s;\n        %s;\n"
            (render_store Isa.sse2 (AZoutTurn (!l, 0)) (render Isa.sse2 tbl lo))
            (render_store Isa.sse2 (AZoutTurn (!l, 1)) (render Isa.sse2 tbl hi))))));
  Buffer.add_string buf "    }\n";
  (* ─── ODD-COUNT TAIL loop (monolithic only): resumes k after the wide
     bulk. `for (; k < count; ++k)` rather than `if` so it generalises when
     per > 2; at per = 2 it runs at most once and predicts perfectly, and a
     count < per call skips the wide loop entirely (the low-trip bypass for
     free). *)
  if not blocked
  then (
    Buffer.add_string
      buf
      "    /* odd-count tail: same DAG at VEX-128, one complex per iteration */\n";
    Buffer.add_string buf "    for (; k < count; ++k) {\n";
    Buffer.add_buffer buf body_n;
    Buffer.add_string buf "    }\n");
  Buffer.add_string buf "}\n";
  Buffer.contents buf
;;

(* ═══════════════════════════════════════════════════════════════
 *  FUSED K=1 — the whole N-point transform as ONE function
 *
 * This is the IL-NATIVE shape, and it is deliberately NOT built by calling
 * the n1t/t2 codelets in sequence. Those carry the split family's staged
 * vocabulary — an 11-arg batched ABI, a memory plane crossed between two
 * function calls, and a RUNTIME twiddle table. At K=1 with N fixed at
 * generation time none of that is wanted:
 *
 *   - MKL's own K=1 interleaved path is ONE function
 *     (docs/research/mkl_highN_cascade_anatomy.md: "the whole 2^k K=1
 *     cascade is ONE function", in-place on a contiguous plane);
 *   - every twiddle is a COMPILE-TIME constant when N is known, so a
 *     runtime table is pure waste — they become file-scope VLITs (CTwV);
 *   - the stage boundary is a register transpose, not an ABI crossing.
 *
 * STRUCTURE — four-step over N = n1*n2, both stages inside one function.
 * Index input as x[j1*n2 + j2] and output as X[k2*n1 + k1].
 *   stage A : for each PAIR of columns (j2, j2+1) — adjacent complexes, so
 *             one vector load per leg — run DFT_n1 over j1 lane-wise. Park
 *             to a function-scope plane P[k1][c].
 *   turn    : two permute2f128 per (k1-pair, c) regroup lanes from
 *             "two j2 of one k1" to "two k1 of one j2" — the four-step
 *             transpose, in REGISTERS.
 *   stage B : multiply by w_N^{k1*j2} (per-lane VLIT, since the two lanes
 *             carry k1 = 2d and 2d+1) and run DFT_n2 over j2. Stores land
 *             at complex k2*n1 + 2d — adjacent, so full-width, and that
 *             address IS natural order. No output permutation, ever.
 * ═══════════════════════════════════════════════════════════════ *)

(* ~chain: the factorization to emit, supplied BY THE CALLER. This emitter
   does NOT choose it.

   Plan selection is the planner's job, decided by MEASURED whole-plan search
   (docs/roadmap/z_chain_planner_notes.md: "DP prunes the search; it never
   composes costs"). An earlier version of this function picked a "squarest
   split" internally — a composed cost model, which both contradicted the
   calibrated chains and caused the measured losses: the N whose split landed
   on spill-free radices (16 = 4x4, 64 = 8x8) BEAT MKL, while those landing on
   r16/r32 (256, 1024) sat at 0.85x. Emitters take the plan as INPUT. *)
let emit_k1
      ~(dir : dir)
      ~(chain_a : int list)
      ~(chain_b : int list)
      ~(isa : Isa.t)
      ~(uarch : Uarch.t)
  : string
  =
  let vw = isa.Isa.vec_width in
  if vw <> 4
  then failwith "codelet_cil: fused K=1 is written for 2 complex/vector (avx2)";
  if chain_a = [] || chain_b = []
  then failwith "codelet_cil: emit_k1 needs a factorization for BOTH passes";
  List.iter
    (fun r ->
       if r < 2 || r > 64 || r land (r - 1) <> 0
       then
         failwith
           (Printf.sprintf
              "codelet_cil: chain factor %d must be a power of two in [2,64]"
              r))
    (chain_a @ chain_b);
  let n1 = List.fold_left ( * ) 1 chain_a
  and n2 = List.fold_left ( * ) 1 chain_b in
  let n = n1 * n2 in
  (* The chain is part of the IDENTITY, not a comment: candidates for one N
     must coexist in one binary to be raced, and the wisdom that names a
     winner has to be able to name it. *)
  let tag_of l = String.concat "x" (List.map string_of_int l) in
  let chain_tag = "a" ^ tag_of chain_a ^ "_b" ^ tag_of chain_b in
  let sign = if dir = Fwd then `Fwd else `Bwd in
  let sgn = if dir = Fwd then -1.0 else 1.0 in
  let pi = 4.0 *. atan 1.0 in
  let tbl : consts = Hashtbl.create 64 in
  let body = Buffer.create 16384 in
  (* VFFT_CX_K1DAG=1 (opt-in): stage B's register transpose becomes DAG data
     — ins are CTurn(CLoad P, CLoad P) nodes, scheduled and rendered like
     everything else. NOT byte-identical to the legacy _a/_b/_t glue (every
     def renames), which is why it is a MODE, gated semantically (bitwise
     output equality) rather than textually. Default = legacy, byte-identical. *)
  let k1dag = Sys.getenv_opt "VFFT_CX_K1DAG" = Some "1" in
  (* One scheduled sub-DAG. Tags restart per call, so each gets its own C
     brace scope; `pre` emits glue (the legacy register transpose) inside it. *)
  let pass
        ~(label : string)
        ~(nin : int)
        ~(pre : unit -> unit)
        ~(lsrc_of : int -> [ `Addr of caddr | `Name of string | `Node of t ])
        ~(build : t array -> t array)
        ~(store : int -> t -> unit)
    : unit
    =
    reset ();
    (* COMPLETE-IR: `Addr inputs are CLoad nodes (stage A — plane loads);
       `Name inputs stay CIn aliases of pre()'s transpose locals (legacy
       stage B); `Node inputs are caller-built subgraphs (K1DAG stage B —
       the in-DAG transpose). Addr/Name keep cin/cload creation order ⇒
       same tags ⇒ byte-identical legacy text. *)
    let ins =
      Array.init nin (fun i ->
        match lsrc_of i with
        | `Addr a -> cload a
        | `Name _ -> cin i
        | `Node e -> e)
    in
    let outs = build ins in
    let assigns = Array.to_list (Array.mapi (fun i e -> Expr.Output (i, true), e) outs) in
    let assigns = Cx_pipeline.prepare_codelet ~who:label ~uarch assigns in
    let sch = cx_schedule uarch assigns in
    Buffer.add_string body (Printf.sprintf "    { /* %s */\n" label);
    pre ();
    (* Addr/Name ins print here (the load edge) and are pre-marked so the
       def walk never reprints them; `Node ins render in the def walk with
       their CLoad children. *)
    let seen : (int, unit) Hashtbl.t = Hashtbl.create 256 in
    Array.iteri
      (fun i (e : t) ->
         match lsrc_of i with
         | `Node _ -> ()
         | `Addr a ->
           Hashtbl.replace seen e.tag ();
           Buffer.add_string
             body
             (Printf.sprintf
                "    %s\n"
                (Isa.const_decl
                   isa
                   (Printf.sprintf "z%d" e.tag)
                   (Isa.loadu_pd isa (addr_str a))))
         | `Name s ->
           Hashtbl.replace seen e.tag ();
           Buffer.add_string
             body
             (Printf.sprintf
                "    %s\n"
                (Isa.const_decl isa (Printf.sprintf "z%d" e.tag) s)))
      ins;
    List.iter
      (fun ((_ : Expr.elem_ref option), (e : t)) ->
         match e.node with
         | CIn _ -> ()
         | _ ->
           if not (Hashtbl.mem seen e.tag)
           then (
             Hashtbl.replace seen e.tag ();
             Buffer.add_string
               body
               (Printf.sprintf
                  "    %s\n"
                  (Isa.const_decl isa (Printf.sprintf "z%d" e.tag) (render isa tbl e)))))
      sch;
    Array.iteri (fun i (e : t) -> store i e) outs;
    Buffer.add_string body "    }\n"
  in
  (* ── stage A: DFT_n1 down each column pair, park to the plane ── *)
  for c = 0 to (n2 / 2) - 1 do
    pass
      ~label:
        (Printf.sprintf "stage A: columns j2=%d,%d -> P[k1][%d]" (2 * c) ((2 * c) + 1) c)
      ~nin:n1
      ~pre:(fun () -> ())
      ~lsrc_of:(fun j1 -> `Addr (AZinAbs (2 * ((j1 * n2) + (2 * c)))))
      ~build:(fun ins -> dft_chain ~sign ~chain:chain_a ins)
      ~store:(fun k1 e ->
        let ad = AP (vw * ((k1 * (n2 / 2)) + c)) in
        let (_ : t) = cstore ad e in
        Buffer.add_string
          body
          (Printf.sprintf
             "    %s;\n"
             (render_store isa ad (Printf.sprintf "z%d" e.tag))))
  done;
  (* ── stage B: register turn + per-lane constant twiddle + DFT_n2 ── *)
  let p2f = Isa.intr isa "permute2f128_pd" in
  for d = 0 to (n1 / 2) - 1 do
    pass
      ~label:
        (Printf.sprintf
           "stage B: k1=%d,%d -> X[k2*%d + %d]"
           (2 * d)
           ((2 * d) + 1)
           n1
           (2 * d))
      ~nin:n2
      ~pre:(fun () ->
        if k1dag then () else
        for c = 0 to (n2 / 2) - 1 do
          Buffer.add_string
            body
            (Printf.sprintf
               "    %s\n    %s\n"
               (Isa.const_decl
                  isa
                  (Printf.sprintf "_a%d" c)
                  (Isa.loadu_pd isa (Printf.sprintf "P[%d]" (vw * ((2 * d * (n2 / 2)) + c)))))
               (Isa.const_decl
                  isa
                  (Printf.sprintf "_b%d" c)
                  (Isa.loadu_pd
                     isa
                     (Printf.sprintf "P[%d]" (vw * ((((2 * d) + 1) * (n2 / 2)) + c))))));
          Buffer.add_string
            body
            (Printf.sprintf
               "    %s\n    %s\n"
               (Isa.const_decl
                  isa
                  (Printf.sprintf "_t%d" (2 * c))
                  (Printf.sprintf "%s(_a%d, _b%d, 0x20)" p2f c c))
               (Isa.const_decl
                  isa
                  (Printf.sprintf "_t%d" ((2 * c) + 1))
                  (Printf.sprintf "%s(_a%d, _b%d, 0x31)" p2f c c)))
        done)
      ~lsrc_of:(fun j2 ->
        if k1dag
        then (
          (* in-DAG register turn: _t{j2} = permute2f128(P-load a, P-load b);
             hash-consing makes the repeated lsrc_of calls hit the same nodes *)
          let c = j2 / 2 in
          let a = cload (AP (vw * ((2 * d * (n2 / 2)) + c)))
          and b = cload (AP (vw * ((((2 * d) + 1) * (n2 / 2)) + c))) in
          `Node (cturn a b (if j2 land 1 = 0 then 0x20 else 0x31)))
        else `Name (Printf.sprintf "_t%d" j2))
      ~build:(fun ins ->
        let tw =
          Array.mapi
            (fun j2 x ->
               if j2 = 0
               then x
               else
                 ctwv
                   (Array.init
                      (vw / 2)
                      (fun lane ->
                         let k1 = (2 * d) + lane in
                         let a =
                           sgn *. 2.0 *. pi *. float_of_int (k1 * j2) /. float_of_int n
                         in
                         cos a, sin a))
                   x)
            ins
        in
        dft_chain ~sign ~chain:chain_b tw)
      ~store:(fun k2 e ->
        let ad = AZoutAbs (2 * ((k2 * n1) + (2 * d))) in
        let (_ : t) = cstore ad e in
        Buffer.add_string
          body
          (Printf.sprintf
             "    %s;\n"
             (render_store isa ad (Printf.sprintf "z%d" e.tag))))
  done;
  let buf = Buffer.create 32768 in
  Buffer.add_string
    buf
    (Printf.sprintf
       "/* Auto-generated by vfft_v2 — FULL-IL FUSED K=1 (codelet_cil.ml).\n\
       \ * N=%d as ONE function: %d x %d four-step, both stages fused, the\n\
       \ * stage boundary is a REGISTER transpose (permute2f128), and every\n\
       \ * twiddle is a compile-time constant — no runtime table, no staged\n\
       \ * codelet ABI, no split conversion anywhere. Natural order in/out.\n\
       \ * Interior is interleaved end to end: %d complex per %d-bit vector. */\n\
        #include <immintrin.h>\n\
        #include <stddef.h>\n\n"
       n
       n1
       n2
       (vw / 2)
       (vw * 64));
  Buffer.add_string
    buf
    (if dir = Fwd
     then Isa.im_mask_decl isa "_M_IM" ^ "  /* x*(-i) */\n"
     else Isa.re_mask_decl isa "_M_RE" ^ "  /* x*(+i) */\n");
  Buffer.add_string buf (emit_const_decls isa tbl);
  Buffer.add_string
    buf
    (Printf.sprintf
       "\n\
        __attribute__((target(\"%s\")))\n\
        void vfft_cil_%d_%s_%s_%s(const double * __restrict__ zin,\n\
       \                              double * __restrict__ zout)\n\
        {\n\
       \    double P[%d];  /* stage-A results; L1-resident, never escapes.\n\
       \                       Flat doubles so &P[i] is the double* that the\n\
       \                       load/store intrinsics take. */\n"
       isa.Isa.target_attr
       n
       chain_tag
       (if dir = Fwd then "fwd" else "bwd")
       isa.Isa.name
       (2 * n));
  Buffer.add_buffer buf body;
  Buffer.add_string buf "}\n";
  Buffer.contents buf
;;
