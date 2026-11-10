(* gen_main.ml — the generator driver: one `run argv` that takes a
 * codelet request from CLI flags to C text on stdout. bin/gen_radix is
 * a 1-line wrapper around it; bin/gen_set calls it in-process once per
 * codelet of the whole tree.
 *
 * Usage (via gen_radix):
 *   gen_radix N                          show DAG and stats (no twiddle)
 *   gen_radix N --emit-c                 emit C code with cost-model defaults
 *
 * Cost-model defaults (auto-applied unless --no-recipe is passed):
 *   Spill (the "full recipe") auto-enables when the cost model says yes.
 *   Currently that's: CT-decomposed AND (n + 6 > vec_regs OR vec_regs >= 32).
 *
 * Override flags:
 *   --no-recipe        force Topo (disable auto spill)
 *   --spill            explicit on (overrides --no-recipe locally per flag)
 *
 * `run` is a single sequential function with five phases, in order:
 *   1. FLAG PARSE — ~50 ref cells, one per CLI flag / family selector.
 *   2. RECIPE — the cost-model auto-defaults above.
 *   3. FAMILY WIRING + MATH DISPATCH — set the Emit_state signature
 *      flags for the selected family, then call the matching
 *      Dft expansion to get raw assignments (+ spill
 *      markers for blocked recipes).
 *   4. PASS CASCADE — Ir.reset, of_assignments, dedup, the
 *      aggressive prime passes, then the FMA cascade with frozen-tag
 *      threading and the marker remap chain. NOTE: pipeline.ml holds
 *      the shared copy of exactly this cascade (used by codelet_oop);
 *      a semantic change here must be mirrored there until the two
 *      are unified.
 *   5. EMIT — the selected family's emitter; without --emit-c, DAG
 *      stats instead.
 * ------------------------------------------------------------------
 * MODULE CARD (gen_main.ml — grep "MODULE CARD" for the full set)
 * ROLE: CLI parse -> recipe -> math layer -> pass cascade -> emit.
 * PIPELINE: the top-level driver; everything else serves it.
 * PUBLIC SURFACE (measured): run — called by bin/gen_radix(1),
 * bin/gen_set(2) and the new-bin clones.
 * DEPS: Dft(31), Algsimp(28), Emit_c(23), Isa(1), Uarch(1).
 * ENV: VFFT_NEWSPLIT, VFFT_FORCE_REASSOC, VFFT_NO_SUBDEDUP,
 * VFFT_DEEP_COLLECT, VFFT_FORCE_FMA_LIFT, VFFT_DISABLE_FMA_LIFT
 * (plus everything the callees read).
 * ------------------------------------------------------------------
 *)

let run (argv : string array) : unit =
  Emit_render.provenance_argv := Some argv;
  (* PER-EMISSION RESET. Every other flag below is a fresh `ref` scoped to
     this call; Cx_math.odd_ct_enabled is module-global, so without this a
     batch emitter (gen_set.exe runs a whole quadrant in ONE process) carries
     --cil-oddct from the first _ct cell into every later odd-radix codelet
     and silently emits the FACTORED form in place of dft_cx_odd's direct
     one. That showed up as 87 odd-radix bodies differing from the shipped
     tree under gen_set while gen_radix -- one process per codelet --
     reproduced them exactly. Reset to the ENV default, not to false, so
     VFFT_CX_ODDCT=1 still works as a global override. *)
  Cx_math.odd_ct_enabled := (Sys.getenv_opt "VFFT_CX_ODDCT" = Some "1");
  let n = ref 4 in
  let emit_c = ref false in
  let spill = ref false in
  let fuse = ref 0 in
  let store_on_compute = ref false in
  let no_recipe = ref false in
  let bwd = ref false in
  (* §12.4 item 3: whole-K=1-four-step mono codelet (N = the positional arg) *)
  let k1_mono = ref false in
  let k1_r1 = ref 0 in
  let k1_il = ref false in
  let k1_sw = ref false in
  (* tier-2 TRUE interleaved-native (z) codelet family (codelet_zil.ml) *)
  let z_native = ref false in
  let z_t2 = ref false in
  let z_blocked = ref false in
  let z_blocked2 = ref false in
  let z_strided = ref false in
  let z_const_tw = ref false in
  let z_pow_tw = ref false in
  let z_tile_ld = ref false in
  let z_pow_tree = ref false in
  let z_strided_st = ref false in
  let z_post_tw = ref false in
  let z_trans_st = ref false in
  (* the block-split stage kernels (cascade_z.ml): the ZTURN-T kinds and
     il2p's msz mids; --zp-<kind>, N (positional) = the radix. *)
  let zp_kind = ref "" in
  (* interleaved-complex (full-IL) family — codelet_cil.ml, §11. Distinct
     from zp_kind so the two emitters stay A/B-able. *)
  let cil_kind = ref "" in
  let cil_rows = ref 1 in
  let cil_rowswap = ref false in      (* --cil-rowswap: the row kind with whole-vector loads and swaps *)
  (* --cil-split m.p : the blocked (2-pass) CT factorization. REQUIRED for a
     non-pow2 blocked emission — the emitter validates it but never invents
     one (a factorization is a plan input, not an emitter decision). *)
  let cil_split = ref "" in
  (* --cil-split3 m.m2.q : the blocked column leaf's THREE-pass factorization *)
  let cil_split3 = ref "" in
  let cil_bwd = ref false in
  let cil_k1 = ref false in
  (* --cil-chain r0.r1: the fused K=1 factorization, supplied by the PLANNER.
     No default: a missing chain is an error, not a heuristic. *)
  let cil_chain = ref "" in
  let cil_blocked = ref false in
  let cil_tangent = ref false in
  let cil_rowloop = ref false in   (* --cil-rowloop: the in-kernel row loop (2026-09-23) *)
  let cil_form_tag = ref false in
  (* --cil-log3: source the T2 mid's VTW2 records sparsely (load the
     power-of-two legs, derive the rest). Full-IL, same table layout. *)
  let cil_log3 = ref false in
  let cil_pretw = ref false in
  let cil_colstride = ref false in
  let cil_gen2 = ref false in
  let cil_grouploop = ref false in
  let cil_transposed = ref false in
  let cil_herm = ref false in
  let cil_st128 = ref false in
  let cil_turnst = ref false in
  let cil_turnst_gs = ref false in
  (* --cil-port P (FP32 only, 2026-11-10): about P percent of the plain data add/sub of the
     DFT body rendered as fmadd(b, 1, a) / fnmadd(b, 1, a) -- bit-identical, moved to the FMA
     ports (Cx_render.port_take) *)
  let cil_port = ref 0 in
  let isa_name = ref "avx512" in
  let uarch_name = ref "sapphire_rapids" in
  let args = Array.to_list argv in
  let i = ref 0 in
  let arr = Array.of_list args in
  while !i < Array.length arr do
    let arg = arr.(!i) in
    if !i = 0
    then ()
    else if arg = "--emit-c"
    then emit_c := true
    else if arg = "--bisect"
    then
      failwith
        "--bisect removed: the GPL-derived bisection port was deleted for license \
         hygiene (2026-07-02); see the REMOVED notice in lib/schedule.ml. Use --su \
         (Starve-Retire) or FFTW upstream for bisection comparisons."
    else if arg = "--spill"
    then spill := true
    else if arg = "--no-recipe"
    then no_recipe := true
    else if arg = "--bwd"
    then bwd := true
    else if arg = "--z-native"
    then z_native := true
    else if arg = "--z-t2"
    then (
      z_native := true;
      z_t2 := true)
    else if arg = "--z-blocked"
    then (
      z_native := true;
      z_blocked := true)
    else if arg = "--z-blocked2"
    then (
      z_native := true;
      z_blocked2 := true)
    else if arg = "--z-t2s"
    then (
      z_native := true;
      z_t2 := true;
      z_strided := true)
    else if arg = "--z-t2c"
    then (
      z_native := true;
      z_t2 := true;
      z_const_tw := true)
    else if arg = "--z-t2sp"
    then (
      z_native := true;
      z_t2 := true;
      z_strided := true;
      z_pow_tw := true)
    else if arg = "--z-t2st"
    then (
      z_native := true;
      z_t2 := true;
      z_strided := true;
      z_tile_ld := true)
    else if arg = "--z-t2spt"
    then (
      z_native := true;
      z_t2 := true;
      z_strided := true;
      z_pow_tw := true;
      z_tile_ld := true)
    else if arg = "--z-t2sq"
    then (
      z_native := true;
      z_t2 := true;
      z_strided := true;
      z_pow_tw := true;
      z_pow_tree := true)
    else if arg = "--z-t2sqt"
    then (
      z_native := true;
      z_t2 := true;
      z_strided := true;
      z_pow_tw := true;
      z_pow_tree := true;
      z_tile_ld := true)
    else if arg = "--zp-msz"
    then zp_kind := "msz"
    else if arg = "--zp-mszb"
    then zp_kind := "mszb"
    else if arg = "--zp-mszt"
    then zp_kind := "mszt"
    else if arg = "--zp-t0tp"
    then zp_kind := "t0tp"
    else if arg = "--zp-t0tpb"
    then zp_kind := "t0tpb"
    else if arg = "--zp-tmg"
    then zp_kind := "tmg"
    else if arg = "--zp-tmgb"
    then zp_kind := "tmgb"
    else if arg = "--zp-tlf"
    then zp_kind := "tlf"
    else if arg = "--zp-tlfb"
    then zp_kind := "tlfb"
    else if arg = "--zp-tlfi"
    then zp_kind := "tlfi"
    else if arg = "--zp-tlfib"
    then zp_kind := "tlfib"
    else if arg = "--zp-t0d"
    then zp_kind := "t0d"
    else if arg = "--zp-tmgd"
    then zp_kind := "tmgd"
    else if arg = "--zp-tld"
    then zp_kind := "tld"
    else if arg = "--zp-tldb"
    then zp_kind := "tldb"
    else if String.length arg > 5 && String.sub arg 0 5 = "--zp-"
            && List.mem (String.sub arg 5 (String.length arg - 5))
                 [ "ts0tp"; "ts0tpb"; "tsmg"; "tsmgb"; "tslf"; "tslfb"; "tslfi"; "tslfib"
                 ; "tv0tp"; "tv0tpb"; "tvmg"; "tvmgb"; "tvlf"; "tvlfb"; "tvlfi"; "tvlfib" ]
    then (* zturn-ts: the split library's ZTURN-T stage kernels (cascade_z.ml) *)
      zp_kind := String.sub arg 5 (String.length arg - 5)
    else if arg = "--cil-split3" && !i + 1 < Array.length arr
    then (
      cil_split3 := arr.(!i + 1);
      incr i)
    else if arg = "--cil-split" && !i + 1 < Array.length arr
    then (
      cil_split := arr.(!i + 1);
      incr i)
    else if arg = "--cil-rows"
    then (
      (* the row kind: rows per iteration (2026-11-08) *)
      (cil_rows :=
         try int_of_string arr.(!i + 1) with
         | _ -> failwith "gen_main: --cil-rows takes an integer");
      incr i)
    else if arg = "--cil-n1"
    then cil_kind := "n1"
    else if arg = "--cil-n1c"
    then cil_kind := "n1c"
    else if arg = "--cil-n1ccs"
    then (
      (* the BATCHED n1c (2026-09-23): n1c + column-stride addressing, lane k =
         ONE WHOLE transform at pitch Gs, two per vector -- the mirror of
         codelet.ml's flag (the corpus round-trips through both) *)
      cil_kind := "n1c";
      cil_colstride := true)
    else if arg = "--cil-row"
    then
      (* the batched ROW kernel (2026-11-08, C2c_il.emit_row): one row per
         iteration, a lane is a point; the successor of --cil-n1ccs *)
      cil_kind := "row"
    else if arg = "--cil-rowswap"
    then (
      (* the row kind, WHOLE-VECTOR LOADS AND SWAPS: lanes = rows, per rows
         per iteration (2026-11-08) *)
      cil_kind := "row";
      cil_rowswap := true)
    else if arg = "--cil-t2c"
    then cil_kind := "t2c"
    else if arg = "--cil-t2csg"
    then (
      (* t2cs + the GENERATED twiddle stream (gen2, 2026-09-04) *)
      cil_kind := "t2";
      cil_colstride := true;
      cil_gen2 := true)
    else if arg = "--cil-t2csgn"
    then (
      (* t2csg + the in-kernel group loop over the natural-base table
         (2026-09-05): the flat DIT's count-1 last stage in one call *)
      cil_kind := "t2";
      cil_colstride := true;
      cil_gen2 := true;
      cil_grouploop := true)
    else if arg = "--cil-t2csgt"
    then (
      (* the TRANSPOSED backward of t2csg (scrambled class, 2026-09-05) *)
      cil_kind := "t2";
      cil_colstride := true;
      cil_gen2 := true;
      cil_transposed := true)
    else if arg = "--cil-t2csgnt"
    then (
      (* the TRANSPOSED backward of t2csgn *)
      cil_kind := "t2";
      cil_colstride := true;
      cil_gen2 := true;
      cil_grouploop := true;
      cil_transposed := true)
    else if arg = "--cil-t2csgh"
    then (
      (* the real flat DIT's Hermitian last stage: t2csgn storing the half spectrum (2026-10-01) *)
      cil_kind := "t2";
      cil_colstride := true;
      cil_gen2 := true;
      cil_grouploop := true;
      cil_herm := true)
    else if arg = "--cil-t2csght"
    then (
      (* its transposed backward: t2csgnt reading the half spectrum *)
      cil_kind := "t2";
      cil_colstride := true;
      cil_gen2 := true;
      cil_grouploop := true;
      cil_transposed := true;
      cil_herm := true)
    else if arg = "--cil-t2cs"
    then (
      (* the column-stride TAIL form of the T2 mid (t2cs, 2026-09-04): the
         flat chain's short-run stages, lanes from two blocks. *)
      cil_kind := "t2";
      cil_colstride := true)
    else if arg = "--cil-t2cp"
    then (
      (* the DELIBERATE, SCOPED re-enable of pre-twiddle placement: t2c at
         fwd only (t2cp, 2026-09-04 — the flat mixed-radix DIT chain's stage
         kind, docs/design/odd_n_cascade_geometry.md). --cil-pretw itself
         stays retired: the t2p bwd twin is not what this is. *)
      cil_kind := "t2c";
      cil_pretw := true)
    else if arg = "--cil-n1t"
    then cil_kind := "n1t"
    else if arg = "--cil-t2h"
    then
      (* the interleaved REAL pair's top stage (real_il.ml, 2026-09-29):
         the Hermitian half-spectrum mid over the stock n1t leaf's plane *)
      cil_kind := "t2h"
    else if arg = "--cil-t2m"
    then cil_kind := "t2m" (* the real pair's top over the real leaf's half spectra *)
    else if arg = "--cil-r2z"
    then cil_kind := "r2z" (* the real pair's real leaf (cx_real.ml) *)
    else if arg = "--cil-rn1"
    then cil_kind := "rn1" (* the real MONO: the whole small real transform as one n1 body *)
    else if arg = "--cil-r1c"
    then cil_kind := "r1c" (* the real FLAT leaf: real legs -> the digit runs *)
    else if arg = "--cil-r2zr"
    then cil_kind := "r2zr" (* the real ROWS: the real leaf over row-major rows *)
    else if arg = "--cil-t2"
    then cil_kind := "t2"
    else if arg = "--cil-bwd"
    then cil_bwd := true
    else if arg = "--cil-k1"
    then cil_k1 := true
    else if arg = "--cil-chain" && !i + 1 < Array.length arr
    then (
      cil_chain := arr.(!i + 1);
      incr i)
    else if arg = "--cil-oddct"
    then
      (* factor odd COMPOSITE radices (9->3x3, 25->5x5, 27->3x9) instead of
         dft_cx_odd's direct form; symbol gains a "ct" tag so both can coexist
         and be raced. Odd primes and every even radix are unaffected. *)
      Cx_math.odd_ct_enabled := true
    else if arg = "--cil-blocked"
    then cil_blocked := true
    else if arg = "--cil-st128"
    then
      (* the n1c column leaf's half-store twin (symbol tag `h`, 2026-10-01):
         the plane stores as two 128-bit halves *)
      cil_st128 := true
    else if arg = "--cil-tangent"
    then cil_tangent := true
    else if arg = "--cil-rowloop"
    then cil_rowloop := true
    else if arg = "--cil-form-tag"
    then cil_form_tag := true
    else if arg = "--cil-log3"
    then
      cil_log3 := true
      (* The store-FORM decoupling the pure-IL INVERSE needs: --cil-turnst
       gives a backward T2 the N1T corner-turned store (symbol tag `t`) —
       t2t, THE canonical bwd flat codelet.
       🔴 --cil-pretw (the t2p kind, PRE-twiddle bwd T2) is RETIRED — Tugbars
       2026-07-29: t2p lost the bwd race at every R1 <= 32 and was deleted
       tree-wide (kernels, registry, plan field, execute route, race arms)
       so no future session re-promotes it by accident. The emitter refuses
       rather than silently emitting an orphan kind. If a pre-twiddle bwd is
       ever genuinely needed, the sanctioned direction is the
       t2t-with-leg-stride store variant (wire OGs), not this flag. *)
    else if arg = "--cil-pretw"
    then
      failwith
        "--cil-pretw (t2p) is RETIRED 2026-07-29 -- t2t is the canonical bwd semantics \
         tree-wide (see src/core/oop/il2p.h). Deliberate re-enable only."
    else if arg = "--cil-turnst"
    then
      cil_turnst := true
      (* --cil-turnst-gs: the T2TG kind (symbol tag `tg`) — t2t's turned store
       with the otherwise-(void)'d OGs argument wired as the LEG STRIDE
       ((leg p, col k) -> zout[2*(k*OLs + p*OGs)], every leg scattered as two
       128-bit halves since strided legs are not contiguous). Built for the
       3-stage odd-chain BACKWARD's middle stage (docs/roadmap/
       il_odd_chain.md: the l' = e + A*f split forces legs at stride A).
       Separate symbol so every existing t2t kernel stays byte-identical. *)
    else if arg = "--cil-turnst-gs"
    then cil_turnst_gs := true
    else if arg = "--cil-port" && !i + 1 < Array.length arr
    then (
      (cil_port :=
         match int_of_string_opt arr.(!i + 1) with
         | Some p when p >= 0 && p <= 100 -> p
         | _ -> failwith "gen_main: --cil-port takes a percentage 0..100");
      incr i)
    else if arg = "--z-t2ss"
    then (
      z_native := true;
      z_t2 := true;
      z_strided := true;
      z_strided_st := true)
    else if arg = "--z-t2d"
    then (
      z_native := true;
      z_t2 := true;
      z_post_tw := true)
    else if arg = "--z-n1t"
    then (
      z_native := true;
      z_trans_st := true)
    else if arg = "--k1-mono"
    then k1_mono := true
    else if arg = "--k1-r1" && !i + 1 < Array.length arr
    then (
      k1_r1 := int_of_string arr.(!i + 1);
      incr i)
    else if arg = "--k1-il"
    then k1_il := true
    else if arg = "--k1-sw"
    then k1_sw := true
    else if arg = "--fuse" && !i + 1 < Array.length arr
    then (
      fuse := int_of_string arr.(!i + 1);
      incr i)
    else if arg = "--store-on-compute"
    then store_on_compute := true
    else if arg = "--isa" && !i + 1 < Array.length arr
    then (
      isa_name := arr.(!i + 1);
      incr i)
    else if arg = "--uarch" && !i + 1 < Array.length arr
    then (
      uarch_name := arr.(!i + 1);
      incr i)
    else (
      try n := int_of_string arg with
      | _ -> ());
    incr i
  done;
  let isa = Isa.of_name !isa_name in
  let uarch = Uarch.of_name !uarch_name in
  if !cil_port <> 0 && not (Isa.is_f32 isa)
  then
    failwith
      "gen_main: --cil-port is an FP32 setting (--isa avx2_f32 / avx512_f32); an FP64 target \
       refuses it";
  if !cil_port <> 0
     && (!cil_k1
         || List.mem !cil_kind [ ""; "row"; "t2h"; "t2m"; "r2z"; "r1c"; "r2zr" ])
  then failwith "gen_main: --cil-port applies to the cil kinds (--cil-n1 / n1c / n1t / t2 / t2c)";
  (* Inform the picker about the target register budget so it can pick
   * ISA-appropriate factorizations (e.g. (4,16) over (8,8) for R=64
   * on AVX2 — narrower live windows fit better in 16 ymm). *)
  Dft_select.target_vec_regs := isa.vec_regs;
  let n = !n in
  let sign = if !bwd then `Bwd else `Fwd in
  (* Cost-model auto-defaults.
   *
   * If --no-recipe is set: don't auto-enable anything.
   * Otherwise: when the rule says yes, turn on --spill automatically.
   *
   * Explicit --spill always takes effect regardless of the rule.
   *
   * Doc 58: the rule now also fires for n1 (no-twiddle) codelets at sizes
   * where blocked construction beats monolithic — see
   * Dft.should_block_n1. Without auto-enable, the n1 path would stay
   * monolithic and miss the spill-recipe wins on the hot path. *)
  let recipe_applicable =
    (not !no_recipe)
    && (Dft.should_block_n1 n isa.vec_regs
        (* NEWSPLIT blocked: E/O1/O3 seam, prototype gate. *)
        || (Sys.getenv_opt "VFFT_NEWSPLIT" = Some "1"
            && (match Dft_select.pick_algorithm n with
                | Dft_select.Split_radix -> true
                | _ -> false)
            && n >= 32))
  in
  if recipe_applicable then if not !spill then spill := true;
  (* M6.1: the per-emission Scratch — created by the DRIVER, one per
     codelet, so nothing leaks across emissions in a warm gen_set process. *)
  let sc = Emit_render.Scratch.create () in
  let raw, spill_markers, spill_ct =
    if
      !spill
      && Sys.getenv_opt "VFFT_NEWSPLIT" = Some "1"
      && (match Dft_select.pick_algorithm n with
          | Dft_select.Split_radix -> true
          | _ -> false)
      && n >= 32
    then (
      let assignments, markers, ct = Dft.dft_expand_newsplit_blocked ~sign n in
      assignments, markers, ct)
    else if !spill && Dft.should_block_n1 n isa.vec_regs
    then (
      (* Doc 58 (updated): n1 codelets at R ≥ 25 use the blocked
       * PASS 1 / PASS 2 structure with spill markers between passes.
       * Required to bound per-pass peak_live and let SU+spill recipe
       * + M-project deliver hand-NFUSE-level vmovapd counts.
       *
       * Smaller R (≤ 16) keeps the monolithic dft_expand path — that
       * regime fits in 32 ZMM registers and blocking is overhead.
       *
       * R=25 was originally in the monolithic bucket on the assumption
       * it "fits in registers", but its 25-element inter-pass live set
       * does NOT fit, and the monolithic emit produces 1128 vector
       * instructions with 450 reg-to-reg moves and 434 stack spills.
       * Switching to blocked emit gave 47% AVX-512 / 39% AVX2 speedup
       * and beats Tugbars's hand-coded 5×5 codelet by 12%. See
       * docs/r25_blocked_emit.md for the analysis. *)
      let assignments, markers, ct = Dft.dft_expand_n1_blocked ~sign n in
      assignments, markers, ct)
    else Dft.dft_expand ~sign n, [], None
  in
  Ir.reset ();
  let policy_n = n in
  let reassoc =
    match Sys.getenv_opt "VFFT_FORCE_REASSOC" with
    | Some "0" -> false
    | Some "1" -> true
    | _ -> Dft_select.needs_reassoc policy_n
  in
  (* M11: the inline cascade DELETED — Pipeline.prepare_codelet is the
     SOLE cascade.  This driver's two historical extra arms travel via the
     recipe (butterfly_share + dup with this driver's uarch and Scratch
     barrier sink); reassoc keeps the env override HERE — caller policy
     (whether family routes adopt it is M11b, a race not a diff). *)
  (* Aggressive prime-only passes: distributive factoring of same-const
   * muls, subsum sharing for X[0] outputs, and Frigo network transposition.
   * For monolithic prime DFTs (R=3/5/7/11) these expose Winograd-style
   * structure invisible to the pair-fold simplifier. CT-decomposed
   * codelets are already in FMA-friendly form — these passes default to
   * no-op for them.
   *
   * For twiddled forms (t1_dit / t1_dif), the inner DFT is wrapped by
   * Cmul nodes; the factor/share passes still help the inner DFT
   * structure. Transposition skips Cmul nodes (they're nonlinear). *)
  let aggressive =
    match Dft_select.pick_algorithm policy_n with
    | Dft_select.Direct -> true
    | Dft_select.Cooley_Tukey _ -> false
    | Dft_select.Split_radix -> false
  in
  let force_fma_lift =
    try Sys.getenv "VFFT_FORCE_FMA_LIFT" = "1" with
    | Not_found -> false
  in
  let disable_fma_lift =
    try Sys.getenv "VFFT_DISABLE_FMA_LIFT" = "1" with
    | Not_found -> false
  in
  let pipe =
    Pipeline.prepare_codelet
      ~recipe:
        { Pipeline.butterfly_share = true
        ; dup =
            Some
              { Pipeline.uarch; barrier_sink = sc.Emit_render.Scratch.dup_barrier_tags }
        }
      ~raw_assigns:raw
      ~spill_markers_raw:spill_markers
      ~spill_ct
      ~reassoc
      ~aggressive
      ~algorithm:(Dft_select.pick_algorithm policy_n)
      ~force_fma_lift
      ~disable_fma_lift
      ~build_spill_info:!spill
      ~fuse:!fuse
  in
  let deduped = pipe.Pipeline.assigns in
  if !emit_c
  then (
    (* MUTUAL EXCLUSION (zil_pipeline_port.md §9 review finding): the pipeline
       zsplit family (the zp flags) and the legacy interleaved-native family
       (the z flags) emit BYTE-IDENTICAL symbol names, and the dispatch below
       is fixed-priority (zp wins). Passing both would silently substitute the
       pipeline kernel under the legacy name — an A/B script that concatenates
       a legacy-z base with a zp variant would then compare pipeline-vs-
       pipeline and report a FALSE parity verdict, exactly in the sterm/sterm2
       family where the per-cell t2q pick turns on plus/minus 5 percent
       placement luck. There is no valid reason to request both in one
       invocation, so fail loudly rather than let the loser arm masquerade as
       the winner. *)
    if (!cil_kind <> "" || !cil_k1) && (!zp_kind <> "" || !z_native || !k1_mono)
    then
      failwith
        "gen_main: --cil-* (interleaved-complex) conflicts with another codelet family \
         in the same invocation; they emit identical symbol names, so pass exactly one \
         family per run.";
    if !zp_kind <> "" && (!z_native || !k1_mono)
    then
      failwith
        (Printf.sprintf
           "gen_main: --zp-%s (pipeline zsplit) conflicts with a legacy \
            --z-*/--k1-mono flag in the same invocation. They emit identical \
            symbol names; pass exactly one family per run."
           !zp_kind);
    if !cil_k1
    then (
      (* FUSED full-IL K=1: the whole N-point transform as ONE function,
         emit-time constant twiddles, register-transpose stage boundary.
         N (positional) = the transform length, not a radix, and the
         factorization comes from --cil-chain (the PLANNER's verdict). *)
      if !cil_chain = ""
      then
        failwith
          "gen_main: --cil-k1 requires --cil-chain A:B — the fused kernel's \
           factorization is a planner decision, and this emitter deliberately has no \
           default (see plans-must-come-from-dp-planner-machinery).";
      (* Syntax: "8.4:8.4" — the colon is the PASS BOUNDARY (the memory plane
         and register turn sit there), and the dot-separated factors on each
         side say how that pass's DFT is itself factored. "32.32" with no colon
         is the degenerate one-radix-per-pass form. Both the split point and
         the interior factorization are plan, so both are spelled out. *)
      let parse_factors what s =
        List.map
          (fun f ->
             match int_of_string_opt f with
             | Some v -> v
             | None ->
               failwith
                 (Printf.sprintf
                    "gen_main: --cil-chain %s: %S is not an integer radix"
                    what
                    f))
          (String.split_on_char '.' s)
      in
      let chain_a, chain_b =
        match String.split_on_char ':' !cil_chain with
        | [ a; b ] -> parse_factors "pass A" a, parse_factors "pass B" b
        | [ one ] ->
          (match parse_factors "chain" one with
           | [ a; b ] -> [ a ], [ b ]
           | _ ->
             failwith
               "gen_main: --cil-chain without a ':' must be exactly two radices (one per \
                pass); use A:B to factor the passes themselves.")
        | _ -> failwith "gen_main: --cil-chain takes at most one ':'"
      in
      let prod = List.fold_left ( * ) 1 (chain_a @ chain_b) in
      if n > 0 && prod <> n
      then
        failwith
          (Printf.sprintf
             "gen_main: --cil-chain %s multiplies to %d but N=%d was requested"
             !cil_chain
             prod
             n);
      print_string
        (C2c_il.emit_k1
           ~tangent:!cil_tangent
           ~dir:(if !cil_bwd then C2c_il.Bwd else C2c_il.Fwd)
           ~chain_a
           ~chain_b
           ~isa
           ~uarch))
    else if !cil_kind = "row"
    then
      print_string
        (C2c_il.emit_row
           ~dir:(if !cil_bwd then C2c_il.Bwd else C2c_il.Fwd)
           ~radix:n
           ~rows:!cil_rows
           ~whole:!cil_rowswap
           ~turnst:!cil_turnst
           ~split:
             (if !cil_split = ""
              then None
              else (
                match String.split_on_char '.' !cil_split with
                | [ a; b ] ->
                  (try Some (int_of_string a, int_of_string b) with
                   | _ -> failwith "gen_main: --cil-split m.p: expected two integers")
                | _ -> failwith "gen_main: --cil-split m.p: expected exactly m.p"))
           ~isa
           ~uarch)
    else if !cil_kind = "t2h" || !cil_kind = "t2m" || !cil_kind = "r2z" || !cil_kind = "r1c" || !cil_kind = "r2zr"
    then
      print_string
        (Real_il.emit
           ~kind:(Real_il.kind_of_string !cil_kind)
           ~dir:(if !cil_bwd then Real_il.Bwd else Real_il.Fwd)
           ~radix:n
           ~isa
           ~uarch)
    else if !cil_kind <> ""
    then
      (* interleaved-complex (full-IL) family: N (positional) = the radix.
         Complex IR -> SHARED SR scheduler (Schedule.Make) -> ISA-parametric
         emission. See codelet_cil.ml. *)
      print_string
        (C2c_il.emit
           ~tangent:!cil_tangent
           ~form_tag:!cil_form_tag
           ~log3:!cil_log3
           ~pretw:!cil_pretw
           ~colstride:!cil_colstride
           ~gen2:!cil_gen2
           ~grouploop:!cil_grouploop
           ~transposed:!cil_transposed
           ~herm:!cil_herm
           ~st128:!cil_st128
           ~rowloop:!cil_rowloop
           ~turnst:!cil_turnst
           ~turnst_gs:!cil_turnst_gs
           ~kind:(C2c_il.kind_of_string !cil_kind)
           ~dir:(if !cil_bwd then C2c_il.Bwd else C2c_il.Fwd)
           ~blocked:!cil_blocked
           ~split:
             (if !cil_split = ""
              then None
              else (
                match String.split_on_char '.' !cil_split with
                | [ a; b ] ->
                  (try Some (int_of_string a, int_of_string b) with
                   | _ ->
                     failwith
                       (Printf.sprintf
                          "gen_main: --cil-split %s: expected two integers m.p"
                          !cil_split))
                | _ ->
                  failwith
                    (Printf.sprintf
                       "gen_main: --cil-split %s: expected exactly m.p"
                       !cil_split)))
           ~split3:
             (if !cil_split3 = ""
              then None
              else (
                match String.split_on_char '.' !cil_split3 with
                | [ a; b; c ] ->
                  (try Some (int_of_string a, int_of_string b, int_of_string c) with
                   | _ ->
                     failwith
                       (Printf.sprintf
                          "gen_main: --cil-split3 %s: expected three integers m.m2.q"
                          !cil_split3))
                | _ ->
                  failwith
                    (Printf.sprintf
                       "gen_main: --cil-split3 %s: expected exactly m.m2.q"
                       !cil_split3)))
           ~radix:n
           ~port:!cil_port
           ~isa
           ~uarch)
    else if !zp_kind <> ""
    then
      (* the block-split stage kernels (cascade_z.ml): N (positional) = the
         radix. Goes through Dft -> Pipeline.prepare_codelet ->
         Schedule.su_schedule -> Emit_c rendering (zil_pipeline_port.md). *)
      print_string
        (Cascade_z.emit_codelet
           ~body_only:false
           ~store_on_compute:!store_on_compute
           ~kind:!zp_kind
           ~radix:n
           ~isa
           ~uarch)
    else if !z_native
    then
      (* RETIRED 2026-07-29. codelet_zil.ml was the self-contained emitter used
         to develop the IL family; every idea in it was re-hosted on the full
         DAG pipeline as codelet_cil.ml (IL kinds) and cascade_z.ml (the
         block-split stage kernels). It bypassed the shared machinery, so it
         had no AVX-512/EPYC path and pass improvements never reached it.
         Use --cil-n1 / --cil-n1t / --cil-t2 (codelet_cil.ml) or the --zp-*
         stage-kernel flags (cascade_z.ml) instead. *)
      failwith
        "--z-* is RETIRED: codelet_zil.ml is gone. Use --cil-{n1,n1t,t2} \
         (codelet_cil.ml) or --zp-* (cascade_z.ml)."
    else if !k1_mono
    then
      (* §12.4 item 3: the whole K=1 four-step as ONE emitted function
         (emit-time rodata twiddles, natural order). N = positional arg. *)
      print_string
        (C2c_split.emit_k1_mono
           ~isa
           ~n
           ~r1_opt:(if !k1_r1 > 0 then Some !k1_r1 else None)
           ~il:!k1_il
           ~sw:!k1_sw))
  else (
    let label = "no-twiddle (n1)" in
    Printf.printf "================================================================\n";
    Printf.printf "  DFT-%d, %s — DAG\n" n label;
    Printf.printf "================================================================\n\n";
    print_string (Algsimp.print_dag deduped);
    Printf.printf "\n================================================================\n";
    Printf.printf "  Stats\n";
    Printf.printf "================================================================\n\n";
    let roots = List.map snd deduped in
    let stats = Algsimp.stats_reachable roots in
    print_string (Algsimp.string_of_stats stats))
;;
