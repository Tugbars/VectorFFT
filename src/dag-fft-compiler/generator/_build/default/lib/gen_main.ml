(* gen_radix.ml — CLI driver.
 *
 * Usage:
 *   gen_radix N                          show DAG and stats (no twiddle)
 *   gen_radix N --twiddled               t1_dit (TP_Flat: load all twiddles)
 *   gen_radix N --twiddled --log3        t1_dit_log3 (load base twiddles, derive rest)
 *   gen_radix N --twiddled --emit-c      emit C code with cost-model defaults
 *   gen_radix N --twiddled --emit-c --bisect   emit with Frigo's bisection scheduler
 *
 * Cost-model defaults (auto-applied unless --no-recipe is passed):
 *   Spill + SU (the "full recipe") auto-enables when the cost model says yes.
 *   Currently that's: CT-decomposed AND (n + 6 > vec_regs OR vec_regs >= 32).
 *
 * Override flags:
 *   --no-recipe        force Topo (disable auto spill+SU)
 *   --spill / --su     explicit on (overrides --no-recipe locally per flag)
 *)

let run (argv : string array) : unit =
  Emit_c.provenance_argv := Some argv;
  let n = ref 4 in
  let twiddled = ref false in
  let twiddled_scalar = ref false in  (* OOP t1s: scalar-broadcast twiddles *)
  let twiddled_pos = ref false in  (* OOP t1p: per-position broadcast twiddles *)
  let log3 = ref false in
  let emit_c = ref false in
  let in_place = ref false in
  let bisect = ref false in
  let annotate = ref false in
  let su = ref false in
  let spill = ref false in
  let fuse = ref 0 in
  let oop_strides = ref None in
  let oop_store_fused = ref false in
  let store_on_compute = ref false in
  let no_recipe = ref false in
  let t1s = ref false in
  let dif = ref false in
  let bwd = ref false in
  let gh = ref false in
  let bb = ref false in
  let bb_budget = ref 1.0 in
  let twidsq = ref false in
  let r2c = ref false in
  let r2c_first = ref false in
  let rdft = ref false in
  let hc2hc = ref false in
  let hc2c = ref false in
  let hc2c_nat = ref false in
  let ranged = ref false in
  let il2 = ref false in
  let r2cf = ref false in
  let r2cb = ref false in
  let r2c_term = ref false in
  let r2c_term_rt = ref false in
  let r2c_term_ls = ref false in
  let r2c_term_ls_r = ref 8 in
  let r2c_term_k = ref 1 in
  let dct2 = ref false in
  let dct2_trigII = ref false in
  let dct3 = ref false in
  let dht = ref false in
  let dst2 = ref false in
  let dst3 = ref false in
  let dct4 = ref false in
  let dst4 = ref false in
  let dct1 = ref false in
  let dst1 = ref false in
  let c2r = ref false in
  let strided = ref false in
  let oop_strided = ref false in  (* strided-OOP n1, pack-fix ABI *)
  (* M2 OOP family CLI args. The --oop flag selects the new codelet
     family (lib/codelet_oop.ml). It's mutually exclusive with --strided
     (the legacy 2D row codelet path) — first one to set wins; we don't
     enforce because each path validates its own preconditions and will
     fail clearly. The four edge-pattern flags drive the new family's
     IR config. *)
  let oop = ref false in
  let oop_load_pat = ref "UL" in   (* "UL" = UnitLeg | "UG" = UnitGroup *)
  let oop_store_pat = ref "UL" in
  let oop_buf_oop = ref false in    (* true → OutOfPlace, false → InPlace *)
  let isa_name = ref "avx512" in
  let uarch_name = ref "sapphire_rapids" in
  let args = Array.to_list argv in
  let i = ref 0 in
  let arr = Array.of_list args in
  while !i < Array.length arr do
    let arg = arr.(!i) in
    (if !i = 0 then ()
     else if arg = "--twiddled" then twiddled := true
     else if arg = "--twiddled-scalar" then (twiddled_scalar := true; twiddled := true)
     else if arg = "--twiddled-pos" then (twiddled_pos := true; twiddled := true)
     else if arg = "--log3"     then log3 := true
     else if arg = "--emit-c"   then emit_c := true
     else if arg = "--in-place" then in_place := true
     else if arg = "--bisect"   then bisect := true
     else if arg = "--annotate" then annotate := true
     else if arg = "--su"       then su := true
     else if arg = "--spill"    then spill := true
     else if arg = "--no-recipe" then no_recipe := true
     else if arg = "--t1s"       then t1s := true
     else if arg = "--dif"       then dif := true
     else if arg = "--bwd"       then bwd := true
     else if arg = "--gh"        then gh := true
     else if arg = "--bb"        then bb := true
     else if arg = "--twidsq"    then twidsq := true
     else if arg = "--r2c"       then r2c := true
     else if arg = "--r2c-first" then r2c_first := true
     else if arg = "--rdft"      then rdft := true
     else if arg = "--hc2hc"     then hc2hc := true
     else if arg = "--hc2c"      then hc2c := true
     else if arg = "--hc2c-nat"  then hc2c_nat := true
     else if arg = "--ranged"    then ranged := true
     else if arg = "--il2"       then il2 := true
     else if arg = "--r2cf"      then r2cf := true
     else if arg = "--r2cb"      then r2cb := true
     else if arg = "--r2c-term"  then r2c_term := true
     else if arg = "--r2c-term-rt" then (r2c_term := true; r2c_term_rt := true)
     else if arg = "--r2c-term-ls" then r2c_term_ls := true
     else if arg = "--dct2"      then dct2 := true
     else if arg = "--dct2-trigII" then dct2_trigII := true
     else if arg = "--dct3"      then dct3 := true
     else if arg = "--dht"       then dht := true
     else if arg = "--dst2"      then dst2 := true
     else if arg = "--dst3"      then dst3 := true
     else if arg = "--dct4"      then dct4 := true
     else if arg = "--dst4"      then dst4 := true
     else if arg = "--dct1"      then dct1 := true
     else if arg = "--dst1"      then dst1 := true
     else if arg = "--c2r"       then c2r := true
     else if arg = "--strided"   then strided := true
     else if arg = "--oop-strided" then oop_strided := true
     else if arg = "--oop"       then oop := true
     else if arg = "--oop-buffer-oop" then oop_buf_oop := true
     else if arg = "--oop-load" && !i + 1 < Array.length arr then begin
       oop_load_pat := arr.(!i + 1);
       incr i
     end
     else if arg = "--oop-store" && !i + 1 < Array.length arr then begin
       oop_store_pat := arr.(!i + 1);
       incr i
     end
     else if arg = "--bb-budget" && !i + 1 < Array.length arr then begin
       bb_budget := float_of_string arr.(!i + 1);
       incr i
     end
     else if arg = "--fuse" && !i + 1 < Array.length arr then begin
       fuse := int_of_string arr.(!i + 1);
       incr i
     end
     else if arg = "--r2c-term-k" && !i + 1 < Array.length arr then begin
       r2c_term_k := int_of_string arr.(!i + 1);
       incr i
     end
     else if arg = "--r2c-term-ls-r" && !i + 1 < Array.length arr then begin
       r2c_term_ls_r := int_of_string arr.(!i + 1);
       incr i
     end
     else if arg = "--oop-strides" && !i + 1 < Array.length arr then begin
       (match String.split_on_char ',' arr.(!i + 1) with
        | [a; b; c; d] ->
          oop_strides := Some (int_of_string a, int_of_string b,
                               int_of_string c, int_of_string d)
        | _ -> failwith "--oop-strides expects L,G,OL,OG (four ints)");
       incr i
     end
     else if arg = "--oop-store-fused" then oop_store_fused := true
     else if arg = "--store-on-compute" then store_on_compute := true
     else if arg = "--isa" && !i + 1 < Array.length arr then begin
       isa_name := arr.(!i + 1);
       incr i
     end
     else if arg = "--uarch" && !i + 1 < Array.length arr then begin
       uarch_name := arr.(!i + 1);
       incr i
     end
     else (try n := int_of_string arg with _ -> ()));
    incr i
  done;
  let isa = Isa.of_name !isa_name in
  let uarch = Uarch.of_name !uarch_name in

  (* Inform the picker about the target register budget so it can pick
   * ISA-appropriate factorizations (e.g. (4,16) over (8,8) for R=64
   * on AVX2 — narrower live windows fit better in 16 ymm). *)
  Dft.target_vec_regs := isa.vec_regs;

  let n = !n in
  let policy : Dft.twiddle_policy =
    if !log3 then TP_Log3 else TP_Flat
  in
  let direction : Dft.direction =
    if !dif then DIF else DIT
  in
  let sign = if !bwd then `Bwd else `Fwd in

  (* Cost-model auto-defaults.
   *
   * If --no-recipe is set: don't auto-enable anything.
   * Otherwise: when the rule says yes AND user didn't override with bisect
   * or annotate, turn on --spill --su automatically.
   *
   * Explicit --spill / --su flags always take effect regardless of the rule.
   *
   * Doc 58: the rule now also fires for n1 (no-twiddle) codelets at sizes
   * where blocked construction beats monolithic — see
   * Dft.should_block_n1. Without auto-enable, the n1 path would stay
   * monolithic and miss the spill-recipe wins on the hot path. *)
  let recipe_applicable =
    not !bisect
    && not !annotate
    && not !no_recipe
    (* strided is single-stage n1-only by design (v1); the blocked
     * recipe emits spill_re/spill_im markers the strided emitter does
     * not declare. Latent since doc-58 auto-blocking; caught when the
     * strided set was first regenerated post-rule (section 38e). *)
    && not !strided
    && (
      (!twiddled && Dft.should_spill n isa.vec_regs)
      (* hc cascade DIT codelets (section 66): the spill variants run
       * the t1 spill builder + output syms; same trigger rule. DIF
       * has no spill route (input-side syms) and stays plain. *)
      || ((!hc2hc || !hc2c || !hc2c_nat) && not !dif
          && Dft.should_spill n isa.vec_regs)
      || (not !twiddled && not !hc2hc && not !hc2c
          && Dft.should_block_n1 n isa.vec_regs)
      (* NEWSPLIT blocked: E/O1/O3 seam, prototype gate. *)
      || (not !twiddled && not !hc2hc && not !hc2c
          && Sys.getenv_opt "VFFT_NEWSPLIT" = Some "1"
          && (match Dft.pick_algorithm n with
              | Dft.Split_radix -> true | _ -> false)
          && n >= 32)
    )
  in
  if recipe_applicable then begin
    if not !spill then spill := true;
    if not !su then su := true
  end;

  (* SU scheduler as the universal default.
   *
   * The recipe above enables it (together with structural spill) for the
   * blocked/spill sizes. This rule also turns it on for the remaining
   * monolithic n1 codelets — the small split-radix / low-n CT sizes
   * (R<=16 n1) that otherwise ship on plain topological scheduling. SU is
   * pressure-aware (Sethi-Ullman load deferral): it shortens live ranges
   * so gcc reuses registers instead of spilling. It is DECOUPLED from
   * --spill here, so these codelets stay monolithic (no cross-pass
   * decomposition is forced on them).
   *
   * Measured, n1 avx2, bit-identical results (0.0e+00):
   *   R8  : 16 -> 6 spill movs, 1.35x faster
   *   R16 : 151 -> 78 spill movs, 0.75x (L1-resident) to 0.98x (memory-bound)
   *   R4  : neutral (0 spills either way)
   * Faster or neutral at every working-set size; no measured regression.
   * The earlier topological default front-loaded loads, which only helps
   * a memory-bound codelet's load parallelism — a case where SU measured
   * neutral, not worse. --no-recipe still selects topological. Codelets
   * that already had SU (all t1s, R>=32 n1) are unaffected (no-op here). *)
  if not !no_recipe && not !bisect && not !annotate && not !su then
    su := true;

  (* Auto-enable Goodman-Hsu mode switch when:
   *   - the recipe is engaged (--su is on), AND
   *   - vec_regs <= 16 (AVX2 or narrower), AND
   *   - n >= 32 (peak live likely to exceed threshold inside clusters)
   *
   * Empirically this delivers 4-8% over the base recipe on AVX2 R={32,64}
   * and is a no-op on AVX-512 (threshold=24 not reached with cluster-sequential).
   * The flag toggle is free at runtime — pressure mode only fires when needed. *)
  if !su && isa.vec_regs <= 16 && n >= 32 && not !no_recipe then begin
    if not !gh then gh := true
  end;

  (* Drive math layer with or without spill marker capture.
   * Spill is meaningful only for twiddled CT-decomposed codelets.
   *
   * --twidsq selects the n×n twidsq DAG (FFTW-style OOP codelet,
   * doc 43): apply inter-stage twiddle, run n parallel DFT-n's,
   * store transposed. Bypasses the regular twiddled / spill paths
   * (those are for in-place codelets). *)
  Emit_c.hoist_consts_enabled :=
    (!rdft || !dct2 || !dct2_trigII || !dct3 || !dst2 || !dst3
     || !dht || !dct4 || !dst4 || !dct1 || !dst1 || !r2cf || !r2cb || !r2c_term || !r2c_term_ls
     || !hc2hc || !hc2c || !hc2c_nat);
  Emit_c.r2cf_signature := !r2cf;
  Emit_c.r2cb_signature := !r2cb;
  Emit_c.r2c_term_signature := !r2c_term;
  Emit_c.r2c_term_rt := !r2c_term_rt;
  Emit_c.r2c_term_laststage := !r2c_term_ls;
  if !r2c_term_ls then Emit_c.r2c_term_ls_r := !r2c_term_ls_r;
  Emit_c.hc_strided := (!hc2hc || !hc2c || !hc2c_nat);
  Emit_c.n1_oop_strided := !oop_strided;
  Emit_c.hc2c_natural := !hc2c_nat;
  Emit_c.hc_ranged := (!ranged && (!hc2hc || !hc2c_nat));
  if !ranged then Emit_c.hc_ranged_r := n;
  if !hc2c_nat then begin
    Emit_c.hc2c_nat_r := n;
    Emit_c.hc2c_nat_sstar :=
      (if n mod 2 = 0 then n / 2 - 1 else (n - 1) / 2)
  end;
  Emit_c.r2r_signature :=
    (!dct2 || !dct2_trigII || !dct3 || !dst2 || !dst3 || !dht
     || !dct4 || !dst4 || !dct1 || !dst1);
  let raw, spill_markers, spill_ct =
    if !r2c then
      (Dft_r2c.dft_expand_r2c ~sign n, [], None)
    else if !r2c_first then
      (Dft_r2c.dft_expand_r2c_first ~sign n, [], None)
    else if !rdft then
      (Dft_r2c.dft_expand_rdft ~sign n, [], None)
    else if !hc2hc then
      let direction = if !dif then `Dif else `Dit in
      if !spill && not !dif then
        Dft_r2c.dft_expand_hc2hc_spill ~sign
          ~tw_policy:(if !log3 then Dft.TP_Log3 else Dft.TP_Flat) n
      else
        (Dft_r2c.dft_expand_hc2hc ~sign ~direction n, [], None)
    else if !r2cf then
      (Dft_r2c.dft_expand_r2cf ~sign n, [], None)
    else if !r2cb then
      (Dft_r2c.dft_expand_r2cb ~sign n, [], None)
    else if !r2c_term then
      (if !r2c_term_rt
       then (Dft_r2c.dft_expand_r2c_term_rt ~sign (), [], None)
       else (Dft_r2c.dft_expand_r2c_term ~sign n !r2c_term_k, [], None))
    else if !r2c_term_ls then
      (let half = n / 2 in let rr = !r2c_term_ls_r in let mm = half / rr in
       (Dft_r2c.dft_expand_r2c_term_laststage ~sign half rr mm, [], None))
    else if !hc2c || !hc2c_nat then
      let direction = if !dif then `Dif else `Dit in
      if !spill && not !dif then
        Dft_r2c.dft_expand_hc2c_spill ~sign
          ~tw_policy:(if !log3 then Dft.TP_Log3 else Dft.TP_Flat) n
      else
        (Dft_r2c.dft_expand_hc2c ~sign ~direction
           ~tw_policy:(if !log3 then Dft.TP_Log3 else Dft.TP_Flat) n, [], None)
    else if !dct2 then
      (Dft_r2c.dft_expand_dct2 n, [], None)
    else if !dct2_trigII then
      (Dft_r2c.dft_expand_dct2_trigII n, [], None)
    else if !dct3 then
      (Dft_r2c.dft_expand_dct3 n, [], None)
    else if !dht then
      (Dft_r2c.dft_expand_dht n, [], None)
    else if !dst2 then
      (Dft_r2c.dft_expand_dst2 n, [], None)
    else if !dst3 then
      (Dft_r2c.dft_expand_dst3 n, [], None)
    else if !dct4 then
      (Dft_r2c.dft_expand_dct4 n, [], None)
    else if !dst4 then
      (Dft_r2c.dft_expand_dst4 n, [], None)
    else if !dct1 then
      (Dft_r2c.dft_expand_dct1 n, [], None)
    else if !dst1 then
      (Dft_r2c.dft_expand_dst1 n, [], None)
    else if !c2r then
      (Dft_r2c.dft_expand_c2r n, [], None)
    else if !twidsq then
      (Dft.dft_expand_twidsq ~direction ~sign n, [], None)
    else if !twiddled && !il2 then
      (Dft.dft_expand_twiddled_il2 ~policy ~direction ~sign n, [], None)
    else if !spill && !twiddled then
      let assignments, markers, ct =
        Dft.dft_expand_twiddled_spill ~policy ~direction ~sign n in
      (assignments, markers, ct)
    else if !twiddled then
      (Dft.dft_expand_twiddled ~policy ~direction ~sign n, [], None)
    else if !spill && not !strided
            && Sys.getenv_opt "VFFT_NEWSPLIT" = Some "1"
            && (match Dft.pick_algorithm n with
                | Dft.Split_radix -> true | _ -> false)
            && n >= 32 then
      let assignments, markers, ct =
        Dft.dft_expand_newsplit_blocked ~sign n in
      (assignments, markers, ct)
    else if !spill && not !strided
            && Dft.should_block_n1 n isa.vec_regs then
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
      let assignments, markers, ct =
        Dft.dft_expand_n1_blocked ~sign n in
      (assignments, markers, ct)
    else
      (Dft.dft_expand ~sign n, [], None)
  in

  Algsimp.reset ();
  (* POLICY SIZE (section 56): simplification policy must follow the
   * DAG, not the CLI n. The boundary kinds wrap an internal rdft of a
   * DIFFERENT size: dct1 builds CT-of-2(N-1), dst1 CT-of-2(N+1). Keying
   * aggressive/reassoc/fma gates on the odd user-facing N classified
   * them as Direct primes and ran the aggressive factor passes over a
   * CT DAG — the documented-unsafe combination (stray same-const fires,
   * algsimp.ml ~1418). Reproducer: dune exec bin/dbg_eval.exe -- 9. *)
  let policy_n =
    if !dct1 then 2 * (n - 1)
    else if !dst1 then 2 * (n + 1)
    else n
  in
  let reassoc =
    match Sys.getenv_opt "VFFT_FORCE_REASSOC" with
    | Some "0" -> false
    | Some "1" -> true
    | _ -> Dft.needs_reassoc policy_n
  in
  let simplified = Algsimp.of_assignments ~reassoc raw in
  let deduped_pre = (if Sys.getenv_opt "VFFT_NO_SUBDEDUP" = Some "1" then (fun x -> x) else Algsimp.dedup_sub_pairs) simplified in

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
  let aggressive = match Dft.pick_algorithm policy_n with
    | Dft.Direct -> true
    | Dft.Cooley_Tukey _ -> false
    | Dft.Split_radix -> false in
  (* For direct primes (n odd prime ≥ 3), the conjugate-pair construction
   * in dft_direct_conjugate_pair already produces optimal hash-cons-shared
   * intermediates (pair sums/diffs, p_re/p_im/q_re/q_im chains). The
   * share_subsums pass — which factors common addition subsums across
   * outputs — actively HURTS this layout: it materializes partial sums
   * that prevent fma_lift from recognizing the unified mixed-sign FMA
   * chain (e.g. it splits `x[0] + c1·s1 + c2·s2 - c3·s3 - c4·s4 - c5·s5`
   * into separate positive and negative sub-chains, costing 4 extra ops
   * per pair output block).
   *
   * Empirically: for R=11, full pipeline gives 172 ops; skipping share
   * gives 150 (matching hand). For R=13, 256 → 204 ops. For composite
   * Cooley-Tukey, share_subsums DOES help (pow2 butterflies have many
   * cross-output partial-sum overlaps), so it's only disabled here for
   * direct-prime mode. *)
  let is_direct = aggressive in  (* aggressive ↔ Direct algorithm in current setup *)
  let factored = Algsimp.factor_common_muls ~aggressive deduped_pre in
  let factored = Algsimp.factor_by_atom ~aggressive factored in
  let factored = (if Sys.getenv_opt "VFFT_NO_SUBDEDUP" = Some "1" then (fun x -> x) else Algsimp.dedup_sub_pairs) factored in
  (* COLLECT-M: opt-in via VFFT_COLLECT_M=1. Walks Add/Sub subtrees and
   * groups terms by their non-constant factor, summing coefficients
   * ("ax + bx + cx -> (a+b+c)x"). Falls through (returns input unchanged)
   * when the env flag is unset. Placed AFTER dedup_sub_pairs so it sees
   * a fully canonicalized form, and BEFORE fma_lift so any new Mul nodes
   * collect_m introduces are visible to FMA fusion. *)
  let factored = Algsimp.collect_m factored in
  (* DEEP-COLLECT: opt-in via VFFT_DEEP_COLLECT=1. Like collect_m but also
   * distributes Const * (Add/Sub) patterns through their inner subtrees
   * (up to a depth limit) to EXPOSE atoms that shallow collect can't see.
   * Use-count-gated: only distributes through subtrees where at least one
   * child has multiple uses (hash-cons-sharing potential). *)
  let factored =
    if Sys.getenv_opt "VFFT_DEEP_COLLECT" = Some "1" then begin
      (* Run deep_collect + collect_m iteratively until fixed point or
       * iteration cap, matching FFTW's fixpoint algsimp. Each iteration
       * may expose new collection opportunities via constant folding
       * introduced in the previous iteration. *)
      let max_iters = 5 in
      let rec loop n cur =
        if n = 0 then cur
        else
          let next = Algsimp.deep_collect cur in
          let next = Algsimp.collect_m next in
          (* Compare by walking root tags. If unchanged, terminate. *)
          let same =
            try
              List.for_all2 (fun (_, a) (_, b) -> a.Algsimp.tag = b.Algsimp.tag)
                cur next
            with Invalid_argument _ -> false
          in
          if same then cur else loop (n - 1) next
      in
      loop max_iters factored
    end
    else factored
  in
  (* Sub(Neg(Mul(a,b)), c) → NK_Fma(a, b, c, true, true) (= fnmsub).
   *
   * Now handled as a peephole inside `mk_sub_binary` so the rewrite fires
   * at construction time (during dedup_sub_pairs' rebuild), not as a
   * standalone post-pass. Standalone post-pass approach orphaned spill
   * marker tags at R=32/R=64; peephole keeps the DAG self-consistent
   * because the rewrite happens before markers get captured.
   * See docs/30_sub_neg_mul_fnmsub.md for the diagnostic that motivated
   * this and docs/31_peephole_vs_post_pass.md for the bug analysis. *)
  let shared =
    if is_direct then factored
    else Algsimp.share_subsums ~aggressive factored
  in
  (* Transposition is correct only when the DAG is purely linear (no Cmul
   * nodes — those wrap symbolic twiddle loads, making the network
   * nonlinear in our representation). For twiddled prime forms (t1_dit,
   * t1_dif), we skip transposition; factor+share still apply to the
   * inner DFT structure. *)
  (* Transpose fixed-point loop removed (dead: gated on
     `aggressive && not is_direct`, always false since aggressive is
     is_direct). Its only consumer of the legacy op-counter goes with
     it, so post_trans is just the shared DAG. *)
  let post_trans = shared in
  (* FMA lift gating — per-codelet-family policy (doc 56).
   *
   * History: doc 28 gated fma_lift to `aggressive` (= Direct primes) only,
   * based on a measurement at R=32 t1_dit_su_spill showing 33-48%
   * regression. That measurement was specific to the UNCONDITIONAL
   * lifting policy (`liftable_mul = true`) which duplicated shared Muls
   * across consumers, blowing op count from 717 to 910 instructions.
   *
   * Doc 56 restored `single_use` as the lifting predicate in
   * algsimp.ml's fma_lift. With single_use:
   *   - Every lift is op-count-preserving (1 mul + 1 add → 1 fma)
   *   - Doc 54's compile failure (hashcons unification across DAG roots)
   *     does not reproduce — that failure mode was driven by duplicated
   *     Mul nodes, not by spill_markers/inline_set as initially diagnosed
   *   - R=32 t1 SU+spill, which doc 28 reported as 33-48% regression,
   *     now WINS 5.2% with single_use fma_lift and up to 26.8% with
   *     fma_lift + M-project combined
   *
   * With those obstacles removed, the gating simplifies to: enable
   * fma_lift everywhere except Split_radix (untested, retain doc 28
   * default).
   *
   * Note: there is a SEPARATE performance question, distinct from
   * correctness — R=64 n1 AVX-512 (the no-twiddle hot path) regresses
   * ~7% with explicit FMA atoms even under single_use. This is not
   * about op counts (266 ops preserved); it's about gcc's freedom to
   * schedule mul+add chains independently of FMA fusion. Resolving
   * this would require either operand-ordering hints in NK_Fma or
   * codelet-specific tuning, both outside doc 56's scope.
   *
   * VFFT_FORCE_FMA_LIFT=1 / VFFT_DISABLE_FMA_LIFT=1 override for
   * experimentation. *)
  let fma_lift_safe =
    match Dft.pick_algorithm policy_n with
    | Dft.Direct -> true
    | Dft.Cooley_Tukey _ -> true
    | Dft.Split_radix -> false in
  let force_fma_lift =
    try Sys.getenv "VFFT_FORCE_FMA_LIFT" = "1" with Not_found -> false in
  let disable_fma_lift =
    try Sys.getenv "VFFT_DISABLE_FMA_LIFT" = "1" with Not_found -> false in
  let apply_fma_lift =
    (fma_lift_safe || force_fma_lift) && not disable_fma_lift in
  (* Capture spill-marker tags BEFORE fma_lift so they can be passed as
   * frozen — fma_lift must not rewrite nodes that are spill targets,
   * because spill_markers reference exact tags and emit_c walks only
   * the reachable-from-assigns subset. Mirror of the doc 30 / doc 31
   * peephole fix for lift_sub_neg_mul.
   *
   * Computing the tags requires running lift_spill_markers on the
   * pre-fma_lift DAG. The same call runs again post-fma_lift to build
   * spill_info; both calls produce the same tags because frozen nodes
   * are preserved by fma_lift. *)
  let frozen_tags =
    if apply_fma_lift && spill_markers <> [] then begin
      let pre_markers = Algsimp.lift_spill_markers ~reassoc spill_markers in
      let tbl = Hashtbl.create 64 in
      List.iter (fun (m : Algsimp.spill_tag_marker) ->
        Hashtbl.replace tbl m.re_tag ();
        Hashtbl.replace tbl m.im_tag ()
      ) pre_markers;
      Some tbl
    end else None
  in
  let deduped =
    if apply_fma_lift then
      Algsimp.fma_lift ~frozen_tags post_trans
    else post_trans
  in
  (* Pass-chain coordination. After a pass returns (new_assigns, remap),
   * its remap's VALUES (new tags) represent the live nodes that
   * previously-frozen spill markers now point to via this remap.
   * Subsequent passes must treat those new tags as frozen, or they'll
   * absorb/rewrite the Muls that are now the spill targets, leaving
   * dead spill references in the output. *)
  let extend_frozen remap =
    match frozen_tags with
    | None -> ()
    | Some tbl ->
      Hashtbl.iter (fun _old_t new_t -> Hashtbl.replace tbl new_t ()) remap
  in
  (* Path A — FFTW-style algebraic factoring of common-const Muls.
   * Recognizes Add(Mul(K,X), Mul(K,Y)) → Mul(K, Add(X,Y)) and similar
   * for Sub. Only fires when factoring is op-neutral or beneficial:
   *  - both Muls' uses are all factor patterns (so the originals die)
   *  - no frozen subtrees touched
   *  - no Add/Sub chain flattening (preserves shared partial-sums)
   *
   * The factored Mul(K, sum) still has multiple consumers downstream;
   * multi_use_fma_lift (next step) absorbs each into an FMA. *)
  let deduped, factor_tag_remap =
    if apply_fma_lift then
      Algsimp.factor_const_muls ~frozen_tags deduped
    else (deduped, Hashtbl.create 0)
  in
  extend_frozen factor_tag_remap;
  (* multi_use_fma_lift: absorbs Muls (including the factored Mul(K, sum)
   * produced by factor_const_muls above) into each consumer Add/Sub
   * as an FMA. Saves 1 op per absorbed Mul and closes the FMA-count
   * gap vs FFTW. *)
  let deduped, mfl_tag_remap =
    if apply_fma_lift then
      Algsimp.multi_use_fma_lift ~frozen_tags deduped
    else (deduped, Hashtbl.create 0)
  in
  extend_frozen mfl_tag_remap;

  (* fma_addend_factor: recognizes Fma(K, X, Mul(K, Y), ...) patterns
   * where the FMA's mul slot and the addend's Mul share the same K.
   * Refactors to Mul(K, X±Y) so the K-multiplication becomes a single
   * outer Mul on a sum/diff, which the follow-up multi_use_fma_lift
   * can then absorb into downstream consumers as FMAs. Closes the
   * "factor across FMA-addend" gap that vanilla factor_const_muls
   * misses (it only sees Add/Sub patterns, not Fma). *)
  (* fma_addend_factor: handles Fma(K, X, Mul(K, Y), ...) and the related
   * Fma(K, X, Neg(Mul(K, Y)), ...) patterns. Closes R=8/16 to FFTW
   * exactly and saves ops at all larger radices. *)
  let deduped, fma_addend_remap =
    if apply_fma_lift then
      Algsimp.fma_addend_factor ~frozen_tags deduped
    else (deduped, Hashtbl.create 0)
  in
  extend_frozen fma_addend_remap;

  (* Second multi_use_fma_lift pass: absorbs new Mul(K, Sum) nodes from
   * fma_addend_factor into their downstream Add/Sub consumers. *)
  let deduped, mfl2_tag_remap =
    if apply_fma_lift then
      Algsimp.multi_use_fma_lift ~frozen_tags deduped
    else (deduped, Hashtbl.create 0)
  in
  extend_frozen mfl2_tag_remap;

  (* Second iteration of fma_addend_factor: the first iteration may
   * produce Neg(Mul) addends in its (nm=true, na=true) case. Those
   * Neg(Mul) addends are themselves factor patterns (type B in the
   * pass — equivalent to Mul with na flipped). A second iteration
   * catches them, and the subsequent mfl3 absorbs the new Muls
   * they produce. *)
  let deduped, fma_addend_remap2 =
    if apply_fma_lift then
      Algsimp.fma_addend_factor ~frozen_tags deduped
    else (deduped, Hashtbl.create 0)
  in
  extend_frozen fma_addend_remap2;

  let deduped, mfl3_tag_remap =
    if apply_fma_lift then
      Algsimp.multi_use_fma_lift ~frozen_tags deduped
    else (deduped, Hashtbl.create 0)
  in
  extend_frozen mfl3_tag_remap;

  (* Third iteration of fma_addend + mfl. Diminishing returns at this
   * point but cheap and harmless if nothing fires. *)
  let deduped, fma_addend_remap3 =
    if apply_fma_lift then
      Algsimp.fma_addend_factor ~frozen_tags deduped
    else (deduped, Hashtbl.create 0)
  in
  extend_frozen fma_addend_remap3;

  let deduped, mfl4_tag_remap =
    if apply_fma_lift then
      Algsimp.multi_use_fma_lift ~frozen_tags deduped
    else (deduped, Hashtbl.create 0)
  in
  extend_frozen mfl4_tag_remap;

  (* flatten_fma_mul_addend: Cat-B finisher. After all the mfl/faf
   * iterations, residual `Fma(A, B, Mul(C, D), nm, na)` patterns whose
   * constants don't match (so fma_addend_factor declined them) still
   * leave a standalone Mul. When such an Fma feeds into an Add/Sub
   * with a third operand P, we can rewrite to a 2-FMA chain
   * `Fma(C, D, Fma(A, B, P, _, _), _, _)`, eliminating the Mul. Saves
   * 1 op per occurrence. See doc 59 (Cat-B section). *)
  let deduped, flatten_tag_remap =
    if apply_fma_lift then
      Algsimp.flatten_fma_mul_addend ~frozen_tags deduped
    else (deduped, Hashtbl.create 0)
  in
  extend_frozen flatten_tag_remap;

  (* Lift spill markers to algsimp tags, then build spill_info.
   * Must happen AFTER of_assignments so hash-consing has run on the
   * marker subtrees.
   *
   * If any of the rewrite passes touched a frozen Add/Sub or Fma node,
   * the spill marker's original tag now points to a dead node. The
   * four remap tables give the new tag with identical algebraic value.
   * Compose them in the order the passes ran. *)
  let spill_info : Emit_c.spill_info option =
    if !spill && spill_markers <> [] then
      let raw_markers = Algsimp.lift_spill_markers ~reassoc spill_markers in
      let remap_tag t =
        let t = match Hashtbl.find_opt factor_tag_remap t with
                | Some t' -> t' | None -> t in
        let t = match Hashtbl.find_opt mfl_tag_remap t with
                | Some t' -> t' | None -> t in
        let t = match Hashtbl.find_opt fma_addend_remap t with
                | Some t' -> t' | None -> t in
        let t = match Hashtbl.find_opt mfl2_tag_remap t with
                | Some t' -> t' | None -> t in
        let t = match Hashtbl.find_opt fma_addend_remap2 t with
                | Some t' -> t' | None -> t in
        let t = match Hashtbl.find_opt mfl3_tag_remap t with
                | Some t' -> t' | None -> t in
        let t = match Hashtbl.find_opt fma_addend_remap3 t with
                | Some t' -> t' | None -> t in
        let t = match Hashtbl.find_opt mfl4_tag_remap t with
                | Some t' -> t' | None -> t in
        t
      in
      let tag_markers = List.map (fun (m : Algsimp.spill_tag_marker) ->
        { m with re_tag = remap_tag m.re_tag; im_tag = remap_tag m.im_tag }
      ) raw_markers in
      Some (Emit_c.make_spill_info ?ct:spill_ct ~fuse:!fuse tag_markers)
    else
      None
  in

  if !emit_c then begin
    (* Symbol naming aligned with production (src/stride-fft/codelets/{isa}/):
     * radix{R}_{variant}_{isa}. No _gen, _inplace, _su, _spill suffixes.
     *
     * The old suffix machinery encoded which codegen passes fired (Sethi-
     * Ullman scheduler, spill recipe, in-place vs OOP). With M-active being
     * universal post-doc-56 and the in-place vs OOP distinction captured
     * by separate variant strings (n1 vs t1_oop), per-symbol decoration of
     * codegen state is noise — the diagnostic info still lives in source
     * comments at the top of each .c. The shorter names match production
     * exactly so prototype codelets drop directly into core symbol slots
     * during the eventual wire-up. *)
    let variant = if !log3 then "_log3" else "" in
    let t1s_infix = if !t1s then "s" else "" in
    let dir_suffix = if !dif then "dif" else "dit" in
    let sgn_suffix = if !bwd then "bwd" else "fwd" in
    let name =
      if !r2c then
        (* R2C forward codelet: radix{N}_r2c_{sgn}_{isa} *)
        Printf.sprintf "radix%d_r2c_%s_%s" n sgn_suffix isa.name
      else if !r2c_first then
        (* R2C first-stage cascade codelet: radix{R}_r2c_first_{sgn}_{isa}
         * The {R} here is the SUB-DFT radix, not the total transform size. *)
        Printf.sprintf "radix%d_r2c_first_%s_%s" n sgn_suffix isa.name
      else if !rdft then
        (* FFTW-style real-input DFT: radix{N}_rdft_{sgn}_{isa} *)
        Printf.sprintf "radix%d_rdft_%s_%s" n sgn_suffix isa.name
      else if !hc2hc && !ranged then
        Printf.sprintf "radix%d_hc2hc_%s%s_rng_%s_%s" n dir_suffix variant sgn_suffix isa.name
      else if !hc2hc then
        (* Middle-stage Hermitian-packed cascade codelet:
         * radix{R}_hc2hc_{dir}{_log3}_{sgn}_{isa} *)
        Printf.sprintf "radix%d_hc2hc_%s%s_%s_%s" n dir_suffix variant sgn_suffix isa.name
      else if !hc2c_nat && !ranged then
        Printf.sprintf "radix%d_hc2c_nat%s_rng_%s_%s" n variant sgn_suffix isa.name
      else if !hc2c_nat then
        (* D2 natural-split terminator (section 69), 4-pointer ABI:
         * radix{R}_hc2c_nat{_log3}_{sgn}_{isa} *)
        Printf.sprintf "radix%d_hc2c_nat%s_%s_%s" n variant sgn_suffix isa.name
      else if !hc2c then
        (* Last-stage cascade codelet: Hermitian-packed in, natural complex out:
         * radix{R}_hc2c_{dir}{_log3}_{sgn}_{isa} *)
        Printf.sprintf "radix%d_hc2c_%s%s_%s_%s" n dir_suffix variant sgn_suffix isa.name
      else if !r2cb then
        (* Native real-cascade BACKWARD leaf (hc2r): radix{R}_r2cb_{isa}.
         * Always backward, so no sgn_suffix. *)
        Printf.sprintf "radix%d_r2cb_%s" n isa.name
      else if !r2cf then
        (* Native real-cascade leaf, stride_n1_fn ABI: radix{R}_r2cf_{isa} *)
        Printf.sprintf "radix%d_r2cf_%s" n isa.name
      else if !r2c_term then
        (* Fused forward terminator (step-2), dual-output ABI.
         * rt variant: frequency-agnostic (runtime twiddle), no k in name. *)
        (if !r2c_term_rt
         then Printf.sprintf "radix%d_r2c_term_rt_%s_%s" n sgn_suffix isa.name
         else Printf.sprintf "radix%d_r2c_term_k%d_%s_%s" n !r2c_term_k sgn_suffix isa.name)
      else if !r2c_term_ls then
        Printf.sprintf "radix%d_r2c_term_ls_r%d_%s_%s" n !r2c_term_ls_r sgn_suffix isa.name
      else if !dct2 then
        (* DCT-II via Makhoul's reduction: radix{N}_dct2_{isa} *)
        Printf.sprintf "radix%d_dct2_%s" n isa.name
      else if !dct2_trigII then
        (* DCT-II via FFTW trigII embedding: radix{N}_dct2_trigII_{isa} *)
        Printf.sprintf "radix%d_dct2_trigII_%s" n isa.name
      else if !dct3 then
        (* DCT-III via inverse-Makhoul: radix{N}_dct3_{isa} *)
        Printf.sprintf "radix%d_dct3_%s" n isa.name
      else if !dht then
        (* DHT (Discrete Hartley Transform): radix{N}_dht_{isa} *)
        Printf.sprintf "radix%d_dht_%s" n isa.name
      else if !dst2 then
        (* DST-II via DCT-II wrapper: radix{N}_dst2_{isa} *)
        Printf.sprintf "radix%d_dst2_%s" n isa.name
      else if !dst3 then
        (* DST-III via DCT-III wrapper: radix{N}_dst3_{isa} *)
        Printf.sprintf "radix%d_dst3_%s" n isa.name
      else if !dct4 then
        (* DCT-IV via Lee 1984: radix{N}_dct4_{isa} *)
        Printf.sprintf "radix%d_dct4_%s" n isa.name
      else if !dst4 then
        (* DST-IV via DCT-IV reduction: radix{N}_dst4_{isa} *)
        Printf.sprintf "radix%d_dst4_%s" n isa.name
      else if !dct1 then
        (* DCT-I via even-extension rdft 2(N-1): radix{N}_dct1_{isa} *)
        Printf.sprintf "radix%d_dct1_%s" n isa.name
      else if !dst1 then
        (* DST-I via odd-extension rdft 2(N+1): radix{N}_dst1_{isa} *)
        Printf.sprintf "radix%d_dst1_%s" n isa.name
      else if !c2r then
        (* C2R backward codelet: radix{N}_c2r_{isa}
         * c2r is always backward, so no separate sgn_suffix is needed. *)
        Printf.sprintf "radix%d_c2r_%s" n isa.name
      else if !twidsq then
        (* Twidsq codelets use their own name pattern reflecting the
         * inter-stage role: radix{N}_twidsq_{dir}_{sgn}_{isa}. *)
        Printf.sprintf "radix%d_twidsq_%s_%s_%s" n dir_suffix sgn_suffix isa.name
      else if !twiddled then
        Printf.sprintf "radix%d_t1%s_%s%s_%s_%s"
          n t1s_infix dir_suffix variant sgn_suffix isa.name
      else
        Printf.sprintf "radix%d_n1_%s_%s%s"
          n sgn_suffix isa.name
          (if !strided then "_strided"
           else if !oop_strided then "_oop_strided" else "")
    in
    let scheduler : Emit_c.scheduler = match !bisect, !su, !annotate with
      | false, false, false -> Topological
      | true,  false, false -> Bisection
      | false, true,  false -> SU uarch
      | false, false, true  -> Annotated_topological
      | true,  false, true  -> Annotated_bisection
      | false, true,  true  -> Annotated_SU uarch
      | _ -> Topological
    in
    let bb_budget_arg = if !bb then Some !bb_budget else None in
    if !oop then begin
      (* M2 OOP codelet family path. The DAG construction inside
         Codelet_oop.emit_codelet is independent of gen_radix's `deduped`
         (it rebuilds the DAG to control the strided=true flag end-to-end);
         we pass only the structural config and the name. The body
         emission path is identical to what Emit_c.emit_codelet does for
         the --strided variant, just with our new edge patterns. *)
      let edge_of_string s = match s with
        | "UL" -> Codelet_oop.UnitLeg
        | "UG" -> Codelet_oop.UnitGroup
        | "SF" -> Codelet_oop.StridedFallback
        | _ -> failwith (Printf.sprintf
                 "--oop-load/--oop-store: unknown pattern %S (expected UL | UG | SF)" s)
      in
      let load_pat  = edge_of_string !oop_load_pat in
      let store_pat = edge_of_string !oop_store_pat in
      let buffer = if !oop_buf_oop
        then Codelet_oop.OutOfPlace
        else Codelet_oop.InPlace in
      let twiddles =
        if !twiddled_pos then Codelet_oop.PerPositionTwiddles
        else if !twiddled_scalar then Codelet_oop.BroadcastTwiddles
        else if !twiddled then Codelet_oop.PerGroupTwiddles
        else Codelet_oop.NoTwiddles in
      let direction = if !bwd
        then Codelet_oop.Backward
        else Codelet_oop.Forward in
      let cname = Codelet_oop.canonical_name
        ~radix:n ~isa ~direction ~load_pat ~store_pat ~buffer ~twiddles in
      let cname = if !log3 then cname ^ "_log3" else cname in
      let cname = if !oop_strides <> None then cname ^ "_spec" else cname in
      let cfg = Codelet_oop.{
        radix = n; isa; direction;
        load_pat; store_pat; buffer; twiddles;
        name = cname;
      } in
      Codelet_oop.current_tw_log3 := !log3;
      Codelet_oop.current_oop_strides := !oop_strides;
      Codelet_oop.current_oop_fuse := !fuse;
      Codelet_oop.current_oop_store_on_compute := !oop_store_fused;
      print_string (Codelet_oop.emit_codelet cfg)
    end else begin
    Emit_c.current_store_on_compute := !store_on_compute;
    print_string (Emit_c.emit_codelet
                    ~in_place:!in_place ~t1s:!t1s ~twidsq:!twidsq
                    ~twidsq_n:(if !twidsq then n else 0)
                    ~strided:!strided ~radix:n
                    ~scheduler ~isa ~gh:!gh
                    ~bb_budget:bb_budget_arg ~spill:spill_info
                    ~is_log3:!log3
                    deduped ~name)
    end
  end else begin
    let variant = if !log3 then ", log3" else "" in
    let label =
      if !twiddled then Printf.sprintf "twiddled (t1_dit%s)" variant
      else "no-twiddle (n1)"
    in
    Printf.printf "================================================================\n";
    Printf.printf "  DFT-%d, %s — DAG\n" n label;
    Printf.printf "================================================================\n\n";
    print_string (Algsimp.print_dag deduped);
    Printf.printf "\n================================================================\n";
    Printf.printf "  Stats\n";
    Printf.printf "================================================================\n\n";
    let roots = List.map snd deduped in
    let stats = Algsimp.stats_reachable roots in
    print_string (Algsimp.string_of_stats stats)
  end