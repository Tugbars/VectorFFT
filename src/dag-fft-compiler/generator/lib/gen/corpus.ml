(* corpus.ml — M10: THE CORPUS, TYPED (§9 #37) — coverage.ml inverted.
 *
 * The family matrices below (moved verbatim from coverage.ml) still
 * CONSTRUCT the per-quadrant (filename, argv_tail) pairs — but they are
 * now the PRIVATE scaffold.  The public surface derives through the
 * Codelet descriptor, with TWO LAWS enforced lazily at first use:
 *
 *   ROUND-TRIP LAW   Codelet.of_argv (tail @ ["--emit-c"]) must
 *                    to_argv back VERBATIM — provenance == coverage ==
 *                    regen recipe becomes a CHECKED fact, one identity
 *                    per codelet (codelet.mli's M5 contract, now load-
 *                    bearing for the tree and the registries).
 *   UNIQUENESS LAW   no two cells share dir/filename; no two cells in
 *                    the whole corpus share a canonical argv.
 *
 * A violation fails LOUDLY in gen_set / the registry emitters — the
 * tree and registry writers refuse to run on a lawless corpus.  The
 * laws never run inside gen_radix itself (lazy: single-codelet
 * emission pays nothing).
 *
 * Consumers: gen_set (tree regen), the six registry emitters (files),
 * emit_registry_h (ip_radices).  ⚠ The emitted registry headers still
 * SAY "Coverage.files" in their comment banner — kept byte-identical
 * deliberately at M10; the wording updates with the first REAL registry
 * change (a coverage raise), never as a rename side effect.
 * cil/zil enter the corpus at M12a (state fix first), never before.
 *)

let ip_radices = [ 2; 3; 4; 5; 6; 7; 8; 10; 11; 12; 13; 16; 17; 19; 20; 25; 32; 64 ]

let oop_n1_radices =
  [ 2; 3; 4; 5; 6; 7; 8; 9; 10; 11; 12; 13; 14; 15; 16; 17; 19; 20; 25; 32; 64; 128 ]
;;

let t1p_radices = [ 4; 7; 8; 13; 16; 32; 64 ]
let spec_radices = [ 7; 13; 32 ] (* stride formula: rv = r * 8 *)
let strided_radices_avx2 = [ 4; 8; 12; 16; 20; 32; 64 ]
let strided_radices_avx512 = [ 8; 16; 32; 64 ]

(* The in-place 18-family matrix: n1 fwd/bwd plus
 * {t1,t1s} x {dit,dif} x {fwd,bwd} x {flat,log3}. *)
let ip_families (isa : string) : (string * string list) list =
  let n1 =
    [ "n1_fwd", [ "--in-place"; "--isa"; isa; "--su" ]
    ; "n1_bwd", [ "--in-place"; "--isa"; isa; "--su"; "--bwd" ]
    ]
  in
  let t1 =
    List.concat_map
      (fun (tsuf, targ) ->
         List.concat_map
           (fun (dsuf, darg) ->
              List.concat_map
                (fun (bsuf, barg) ->
                   List.map
                     (fun (lsuf, larg) ->
                        let fam = tsuf ^ dsuf ^ bsuf ^ lsuf in
                        let args =
                          [ "--twiddled"; "--in-place"; "--isa"; isa ]
                          @ targ
                          @ darg
                          @ barg
                          @ larg
                        in
                        fam, args)
                     [ "", []; "_log3", [ "--log3" ] ])
                [ "_fwd", []; "_bwd", [ "--bwd" ] ])
           [ "_dit", []; "_dif", [ "--dif" ] ])
      [ "t1", []; "t1s", [ "--t1s" ] ]
  in
  n1 @ t1
;;

let oop_base (isa : string) : string list =
  [ "--oop"; "--oop-buffer-oop"; "--oop-load"; "UG"; "--oop-store"; "UG"; "--isa"; isa ]
;;

(* (filename, argv_tail) pairs per quadrant. argv_tail excludes the exe
 * name and --emit-c. Filenames match the committed tree exactly. *)
(* ── M12a: the cil/zsplit quadrants ──
   LITERAL cells, not a comprehension: the pure-IL and boundary-split sets
   are BENCHMARK-DERIVED (the derive arms) — irregular by nature.  Derived
   once from gates/recipes.tsv (recorded provenance), restricted to cells
   that reproduce IDENTICAL with no env/sed: the 6 stale pure_il cells,
   the 6 tangent sunset copies and the 9 sed-renamed replay rows stay OUT
   — regenerating those would overwrite shipped bytes.  The corpus laws
   check every row at first use; the entry was gated by a WARM gen_set regen
   to a temp root: 221/253 byte-identical, 32/32 boundary cells
   BODY-identical with only the argv[0] token in the provenance line
   differing (the shipped 32 recorded a full WSL path; gen_set stamps the
   logical name "gen_radix.exe" â the KNOWN 123-file hermeticity item,
   Â§14.3, queued for its own announced regen).  A gen_set rewrite of
   zil-boundary before that regen shows 32 prologue-only diffs in git â
   that is THIS item, never silent drift. *)
let zil_boundary_cells : (string * string list) list =
  [ ( "radix4_z_dts_r4_avx2.c"
    , [ "4"; "--zp-dts"; "--zp-r0"; "4"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ]
    )
  ; ( "radix4_z_dtsn_r4_avx2.c"
    , [ "4"; "--zp-dtsn"; "--zp-r0"; "4"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ]
    )
  ; ( "radix4_z_dtso_r4_avx2.c"
    , [ "4"; "--zp-dtso"; "--zp-r0"; "4"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ]
    )
  ; ( "radix4_z_dtt_r4_avx2.c"
    , [ "4"; "--zp-dtt"; "--zp-r0"; "4"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ]
    )
  ; ( "radix4_z_msd_avx2.c"
    , [ "4"; "--zp-msd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix4_z_msg_avx2.c"
    , [ "4"; "--zp-msg"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix4_z_msg_bwd_avx2.c"
    , [ "4"; "--zp-msgb"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix4_z_s0s_avx2.c"
    , [ "4"; "--zp-s0s"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix4_z_s0s_bwd_avx2.c"
    , [ "4"; "--zp-s0sb"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix4_z_s0t_r4_avx2.c"
    , [ "4"; "--zp-s0t"; "--zp-r0"; "4"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ]
    )
  ; ( "radix4_z_s0t_r4_bwd_avx2.c"
    , [ "4"; "--zp-s0tb"; "--zp-r0"; "4"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ]
    )
  ; ( "radix4_z_stf_r4_avx2.c"
    , [ "4"; "--zp-stf"; "--zp-r0"; "4"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ]
    )
  ; ( "radix4_z_stf_r4_bwd_avx2.c"
    , [ "4"; "--zp-stfb"; "--zp-r0"; "4"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ]
    )
  ; ( "radix4_z_stfn_r4_avx2.c"
    , [ "4"; "--zp-stfn"; "--zp-r0"; "4"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ]
    )
  ; ( "radix4_z_stfn_r4_bwd_avx2.c"
    , [ "4"
      ; "--zp-stfbn"
      ; "--zp-r0"
      ; "4"
      ; "--isa"
      ; "avx2"
      ; "--uarch"
      ; "raptor_lake_avx2"
      ] )
  ; ( "radix8_z_dts_r4_avx2.c"
    , [ "8"; "--zp-dts"; "--zp-r0"; "4"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ]
    )
  ; ( "radix8_z_dtsn_r4_avx2.c"
    , [ "8"; "--zp-dtsn"; "--zp-r0"; "4"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ]
    )
  ; ( "radix8_z_dtso_r4_avx2.c"
    , [ "8"; "--zp-dtso"; "--zp-r0"; "4"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ]
    )
  ; ( "radix8_z_msd_avx2.c"
    , [ "8"; "--zp-msd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix8_z_msg_avx2.c"
    , [ "8"; "--zp-msg"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix8_z_msg_bwd_avx2.c"
    , [ "8"; "--zp-msgb"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix8_z_s0s_avx2.c"
    , [ "8"; "--zp-s0s"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix8_z_s0s_bwd_avx2.c"
    , [ "8"; "--zp-s0sb"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix8_z_sterm2_avx2.c"
    , [ "8"; "--zp-sterm2"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix8_z_sterm_avx2.c"
    , [ "8"; "--zp-sterm"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix8_z_sterm_bwd_avx2.c"
    , [ "8"; "--zp-stermb"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix8_z_stf2_r4_avx2.c"
    , [ "8"; "--zp-stf2"; "--zp-r0"; "4"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ]
    )
  ; ( "radix8_z_stf_r4_avx2.c"
    , [ "8"; "--zp-stf"; "--zp-r0"; "4"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ]
    )
  ; ( "radix8_z_stf_r4_bwd_avx2.c"
    , [ "8"; "--zp-stfb"; "--zp-r0"; "4"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ]
    )
  ; ( "radix8_z_stf_r4sk_bwd_avx2.c"
    , [ "8"
      ; "--zp-stfb"
      ; "--zp-r0"
      ; "4"
      ; "--zp-sink"
      ; "--isa"
      ; "avx2"
      ; "--uarch"
      ; "raptor_lake_avx2"
      ] )
  ; ( "radix8_z_stfn_r4_avx2.c"
    , [ "8"; "--zp-stfn"; "--zp-r0"; "4"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ]
    )
  ; ( "radix8_z_stfn_r4_bwd_avx2.c"
    , [ "8"
      ; "--zp-stfbn"
      ; "--zp-r0"
      ; "4"
      ; "--isa"
      ; "avx2"
      ; "--uarch"
      ; "raptor_lake_avx2"
      ] )
  ]
;;

let zil_pure_cells : (string * string list) list =
  [ ( "radix10_z_n1_avx2.c"
    , [ "10"; "--cil-n1"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix10_z_n1_bwd_avx2.c"
    , [ "10"; "--cil-n1"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix10_z_n1t_avx2.c"
    , [ "10"; "--cil-n1t"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix10_z_n1t_bwd_avx2.c"
    , [ "10"; "--cil-n1t"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix10_z_t2_avx2.c"
    , [ "10"; "--cil-t2"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix10_z_t2_bwd_avx2.c"
    , [ "10"; "--cil-t2"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix10_z_t2t_bwd_avx2.c"
    , [ "10"
      ; "--cil-t2"
      ; "--cil-turnst"
      ; "--cil-bwd"
      ; "--isa"
      ; "avx2"
      ; "--uarch"
      ; "raptor_lake_avx2"
      ] )
  ; ( "radix10_z_t2tg_bwd_avx2.c"
    , [ "10"
      ; "--cil-t2"
      ; "--cil-turnst-gs"
      ; "--cil-bwd"
      ; "--isa"
      ; "avx2"
      ; "--uarch"
      ; "raptor_lake_avx2"
      ] )
  ; ( "radix11_z_n1_avx2.c"
    , [ "11"; "--cil-n1"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix11_z_n1_bwd_avx2.c"
    , [ "11"; "--cil-n1"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix11_z_n1t_avx2.c"
    , [ "11"; "--cil-n1t"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix11_z_n1t_bwd_avx2.c"
    , [ "11"; "--cil-n1t"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix11_z_t2_avx2.c"
    , [ "11"; "--cil-t2"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix11_z_t2_bwd_avx2.c"
    , [ "11"; "--cil-t2"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix11_z_t2_log3_avx2.c"
    , [ "11"; "--cil-t2"; "--cil-log3"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix11_z_t2_log3_bwd_avx2.c"
    , [ "11"
      ; "--cil-t2"
      ; "--cil-log3"
      ; "--cil-bwd"
      ; "--isa"
      ; "avx2"
      ; "--uarch"
      ; "raptor_lake_avx2"
      ] )
  ; ( "radix11_z_t2t_bwd_avx2.c"
    , [ "11"
      ; "--cil-t2"
      ; "--cil-turnst"
      ; "--cil-bwd"
      ; "--isa"
      ; "avx2"
      ; "--uarch"
      ; "raptor_lake_avx2"
      ] )
  ; ( "radix11_z_t2tg_bwd_avx2.c"
    , [ "11"
      ; "--cil-t2"
      ; "--cil-turnst-gs"
      ; "--cil-bwd"
      ; "--isa"
      ; "avx2"
      ; "--uarch"
      ; "raptor_lake_avx2"
      ] )
  ; ( "radix12_z_n1_avx2.c"
    , [ "12"; "--cil-n1"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix12_z_n1_bwd_avx2.c"
    , [ "12"; "--cil-n1"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix12_z_n1t_avx2.c"
    , [ "12"; "--cil-n1t"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix12_z_n1t_bwd_avx2.c"
    , [ "12"; "--cil-n1t"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix12_z_t2_avx2.c"
    , [ "12"; "--cil-t2"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix12_z_t2_bwd_avx2.c"
    , [ "12"; "--cil-t2"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix12_z_t2t_bwd_avx2.c"
    , [ "12"
      ; "--cil-t2"
      ; "--cil-turnst"
      ; "--cil-bwd"
      ; "--isa"
      ; "avx2"
      ; "--uarch"
      ; "raptor_lake_avx2"
      ] )
  ; ( "radix12_z_t2tg_bwd_avx2.c"
    , [ "12"
      ; "--cil-t2"
      ; "--cil-turnst-gs"
      ; "--cil-bwd"
      ; "--isa"
      ; "avx2"
      ; "--uarch"
      ; "raptor_lake_avx2"
      ] )
  ; ( "radix13_z_n1_avx2.c"
    , [ "13"; "--cil-n1"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix13_z_n1_bwd_avx2.c"
    , [ "13"; "--cil-n1"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix13_z_n1t_avx2.c"
    , [ "13"; "--cil-n1t"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix13_z_n1t_bwd_avx2.c"
    , [ "13"; "--cil-n1t"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix13_z_t2_avx2.c"
    , [ "13"; "--cil-t2"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix13_z_t2_bwd_avx2.c"
    , [ "13"; "--cil-t2"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix13_z_t2_log3_avx2.c"
    , [ "13"; "--cil-t2"; "--cil-log3"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix13_z_t2_log3_bwd_avx2.c"
    , [ "13"
      ; "--cil-t2"
      ; "--cil-log3"
      ; "--cil-bwd"
      ; "--isa"
      ; "avx2"
      ; "--uarch"
      ; "raptor_lake_avx2"
      ] )
  ; ( "radix13_z_t2t_bwd_avx2.c"
    , [ "13"
      ; "--cil-t2"
      ; "--cil-turnst"
      ; "--cil-bwd"
      ; "--isa"
      ; "avx2"
      ; "--uarch"
      ; "raptor_lake_avx2"
      ] )
  ; ( "radix13_z_t2tg_bwd_avx2.c"
    , [ "13"
      ; "--cil-t2"
      ; "--cil-turnst-gs"
      ; "--cil-bwd"
      ; "--isa"
      ; "avx2"
      ; "--uarch"
      ; "raptor_lake_avx2"
      ] )
  ; ( "radix15_z_n1_avx2.c"
    , [ "15"; "--cil-n1"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix15_z_n1_bwd_avx2.c"
    , [ "15"; "--cil-n1"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix15_z_n1b_avx2.c"
    , [ "15"
      ; "--cil-n1"
      ; "--cil-blocked"
      ; "--cil-split"
      ; "3.5"
      ; "--isa"
      ; "avx2"
      ; "--uarch"
      ; "raptor_lake_avx2"
      ] )
  ; ( "radix15_z_n1b_bwd_avx2.c"
    , [ "15"
      ; "--cil-n1"
      ; "--cil-blocked"
      ; "--cil-split"
      ; "3.5"
      ; "--cil-bwd"
      ; "--isa"
      ; "avx2"
      ; "--uarch"
      ; "raptor_lake_avx2"
      ] )
  ; ( "radix15_z_n1t_avx2.c"
    , [ "15"; "--cil-n1t"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix15_z_n1t_bwd_avx2.c"
    , [ "15"; "--cil-n1t"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix15_z_t2_avx2.c"
    , [ "15"; "--cil-t2"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix15_z_t2_bwd_avx2.c"
    , [ "15"; "--cil-t2"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix15_z_t2_log3_avx2.c"
    , [ "15"; "--cil-t2"; "--cil-log3"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix15_z_t2_log3_bwd_avx2.c"
    , [ "15"
      ; "--cil-t2"
      ; "--cil-log3"
      ; "--cil-bwd"
      ; "--isa"
      ; "avx2"
      ; "--uarch"
      ; "raptor_lake_avx2"
      ] )
  ; ( "radix15_z_t2b_avx2.c"
    , [ "15"
      ; "--cil-t2"
      ; "--cil-blocked"
      ; "--cil-split"
      ; "3.5"
      ; "--isa"
      ; "avx2"
      ; "--uarch"
      ; "raptor_lake_avx2"
      ] )
  ; ( "radix15_z_t2b_bwd_avx2.c"
    , [ "15"
      ; "--cil-t2"
      ; "--cil-blocked"
      ; "--cil-split"
      ; "3.5"
      ; "--cil-bwd"
      ; "--isa"
      ; "avx2"
      ; "--uarch"
      ; "raptor_lake_avx2"
      ] )
  ; ( "radix15_z_t2b_log3_avx2.c"
    , [ "15"
      ; "--cil-t2"
      ; "--cil-blocked"
      ; "--cil-split"
      ; "3.5"
      ; "--cil-log3"
      ; "--isa"
      ; "avx2"
      ; "--uarch"
      ; "raptor_lake_avx2"
      ] )
  ; ( "radix15_z_t2b_log3_bwd_avx2.c"
    , [ "15"
      ; "--cil-t2"
      ; "--cil-blocked"
      ; "--cil-split"
      ; "3.5"
      ; "--cil-log3"
      ; "--cil-bwd"
      ; "--isa"
      ; "avx2"
      ; "--uarch"
      ; "raptor_lake_avx2"
      ] )
  ; ( "radix15_z_t2t_bwd_avx2.c"
    , [ "15"
      ; "--cil-t2"
      ; "--cil-turnst"
      ; "--cil-bwd"
      ; "--isa"
      ; "avx2"
      ; "--uarch"
      ; "raptor_lake_avx2"
      ] )
  ; ( "radix15_z_t2tg_bwd_avx2.c"
    , [ "15"
      ; "--cil-t2"
      ; "--cil-turnst-gs"
      ; "--cil-bwd"
      ; "--isa"
      ; "avx2"
      ; "--uarch"
      ; "raptor_lake_avx2"
      ] )
  ; ( "radix16_z_n1_bwd_avx2.c"
    , [ "16"; "--cil-n1"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix16_z_n1t_avx2.c"
    , [ "16"; "--cil-n1t"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix16_z_n1t_bwd_avx2.c"
    , [ "16"; "--cil-n1t"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix16_z_t2_avx2.c"
    , [ "16"; "--cil-t2"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix16_z_t2_bwd_avx2.c"
    , [ "16"; "--cil-t2"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix16_z_t2t_bwd_avx2.c"
    , [ "16"
      ; "--cil-t2"
      ; "--cil-turnst"
      ; "--cil-bwd"
      ; "--isa"
      ; "avx2"
      ; "--uarch"
      ; "raptor_lake_avx2"
      ] )
  ; ( "radix16_z_t2tg_bwd_avx2.c"
    , [ "16"
      ; "--cil-t2"
      ; "--cil-turnst-gs"
      ; "--cil-bwd"
      ; "--isa"
      ; "avx2"
      ; "--uarch"
      ; "raptor_lake_avx2"
      ] )
  ; ( "radix17_z_n1_avx2.c"
    , [ "17"; "--cil-n1"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix17_z_n1_bwd_avx2.c"
    , [ "17"; "--cil-n1"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix17_z_n1t_avx2.c"
    , [ "17"; "--cil-n1t"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix17_z_n1t_bwd_avx2.c"
    , [ "17"; "--cil-n1t"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix17_z_t2_avx2.c"
    , [ "17"; "--cil-t2"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix17_z_t2_bwd_avx2.c"
    , [ "17"; "--cil-t2"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix17_z_t2_log3_avx2.c"
    , [ "17"; "--cil-t2"; "--cil-log3"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix17_z_t2_log3_bwd_avx2.c"
    , [ "17"
      ; "--cil-t2"
      ; "--cil-log3"
      ; "--cil-bwd"
      ; "--isa"
      ; "avx2"
      ; "--uarch"
      ; "raptor_lake_avx2"
      ] )
  ; ( "radix17_z_t2t_bwd_avx2.c"
    , [ "17"
      ; "--cil-t2"
      ; "--cil-turnst"
      ; "--cil-bwd"
      ; "--isa"
      ; "avx2"
      ; "--uarch"
      ; "raptor_lake_avx2"
      ] )
  ; ( "radix17_z_t2tg_bwd_avx2.c"
    , [ "17"
      ; "--cil-t2"
      ; "--cil-turnst-gs"
      ; "--cil-bwd"
      ; "--isa"
      ; "avx2"
      ; "--uarch"
      ; "raptor_lake_avx2"
      ] )
  ; ( "radix19_z_n1_avx2.c"
    , [ "19"; "--cil-n1"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix19_z_n1_bwd_avx2.c"
    , [ "19"; "--cil-n1"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix19_z_n1t_avx2.c"
    , [ "19"; "--cil-n1t"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix19_z_n1t_bwd_avx2.c"
    , [ "19"; "--cil-n1t"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix19_z_t2_avx2.c"
    , [ "19"; "--cil-t2"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix19_z_t2_bwd_avx2.c"
    , [ "19"; "--cil-t2"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix19_z_t2_log3_avx2.c"
    , [ "19"; "--cil-t2"; "--cil-log3"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix19_z_t2_log3_bwd_avx2.c"
    , [ "19"
      ; "--cil-t2"
      ; "--cil-log3"
      ; "--cil-bwd"
      ; "--isa"
      ; "avx2"
      ; "--uarch"
      ; "raptor_lake_avx2"
      ] )
  ; ( "radix19_z_t2t_bwd_avx2.c"
    , [ "19"
      ; "--cil-t2"
      ; "--cil-turnst"
      ; "--cil-bwd"
      ; "--isa"
      ; "avx2"
      ; "--uarch"
      ; "raptor_lake_avx2"
      ] )
  ; ( "radix19_z_t2tg_bwd_avx2.c"
    , [ "19"
      ; "--cil-t2"
      ; "--cil-turnst-gs"
      ; "--cil-bwd"
      ; "--isa"
      ; "avx2"
      ; "--uarch"
      ; "raptor_lake_avx2"
      ] )
  ; ( "radix21_z_n1_avx2.c"
    , [ "21"; "--cil-n1"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix21_z_n1_bwd_avx2.c"
    , [ "21"; "--cil-n1"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix21_z_n1b_avx2.c"
    , [ "21"
      ; "--cil-n1"
      ; "--cil-blocked"
      ; "--cil-split"
      ; "3.7"
      ; "--isa"
      ; "avx2"
      ; "--uarch"
      ; "raptor_lake_avx2"
      ] )
  ; ( "radix21_z_n1b_bwd_avx2.c"
    , [ "21"
      ; "--cil-n1"
      ; "--cil-blocked"
      ; "--cil-split"
      ; "3.7"
      ; "--cil-bwd"
      ; "--isa"
      ; "avx2"
      ; "--uarch"
      ; "raptor_lake_avx2"
      ] )
  ; ( "radix21_z_n1t_avx2.c"
    , [ "21"; "--cil-n1t"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix21_z_n1t_bwd_avx2.c"
    , [ "21"; "--cil-n1t"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix21_z_t2_avx2.c"
    , [ "21"; "--cil-t2"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix21_z_t2_bwd_avx2.c"
    , [ "21"; "--cil-t2"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix21_z_t2_log3_avx2.c"
    , [ "21"; "--cil-t2"; "--cil-log3"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix21_z_t2_log3_bwd_avx2.c"
    , [ "21"
      ; "--cil-t2"
      ; "--cil-log3"
      ; "--cil-bwd"
      ; "--isa"
      ; "avx2"
      ; "--uarch"
      ; "raptor_lake_avx2"
      ] )
  ; ( "radix21_z_t2b_avx2.c"
    , [ "21"
      ; "--cil-t2"
      ; "--cil-blocked"
      ; "--cil-split"
      ; "3.7"
      ; "--isa"
      ; "avx2"
      ; "--uarch"
      ; "raptor_lake_avx2"
      ] )
  ; ( "radix21_z_t2b_bwd_avx2.c"
    , [ "21"
      ; "--cil-t2"
      ; "--cil-blocked"
      ; "--cil-split"
      ; "3.7"
      ; "--cil-bwd"
      ; "--isa"
      ; "avx2"
      ; "--uarch"
      ; "raptor_lake_avx2"
      ] )
  ; ( "radix21_z_t2b_log3_avx2.c"
    , [ "21"
      ; "--cil-t2"
      ; "--cil-blocked"
      ; "--cil-split"
      ; "3.7"
      ; "--cil-log3"
      ; "--isa"
      ; "avx2"
      ; "--uarch"
      ; "raptor_lake_avx2"
      ] )
  ; ( "radix21_z_t2b_log3_bwd_avx2.c"
    , [ "21"
      ; "--cil-t2"
      ; "--cil-blocked"
      ; "--cil-split"
      ; "3.7"
      ; "--cil-log3"
      ; "--cil-bwd"
      ; "--isa"
      ; "avx2"
      ; "--uarch"
      ; "raptor_lake_avx2"
      ] )
  ; ( "radix21_z_t2t_bwd_avx2.c"
    , [ "21"
      ; "--cil-t2"
      ; "--cil-turnst"
      ; "--cil-bwd"
      ; "--isa"
      ; "avx2"
      ; "--uarch"
      ; "raptor_lake_avx2"
      ] )
  ; ( "radix21_z_t2tg_bwd_avx2.c"
    , [ "21"
      ; "--cil-t2"
      ; "--cil-turnst-gs"
      ; "--cil-bwd"
      ; "--isa"
      ; "avx2"
      ; "--uarch"
      ; "raptor_lake_avx2"
      ] )
  ; ( "radix25_z_n1_avx2.c"
    , [ "25"; "--cil-n1"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix25_z_n1_bwd_avx2.c"
    , [ "25"; "--cil-n1"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix25_z_n1b_avx2.c"
    , [ "25"
      ; "--cil-n1"
      ; "--cil-blocked"
      ; "--cil-split"
      ; "5.5"
      ; "--isa"
      ; "avx2"
      ; "--uarch"
      ; "raptor_lake_avx2"
      ] )
  ; ( "radix25_z_n1b_bwd_avx2.c"
    , [ "25"
      ; "--cil-n1"
      ; "--cil-blocked"
      ; "--cil-split"
      ; "5.5"
      ; "--cil-bwd"
      ; "--isa"
      ; "avx2"
      ; "--uarch"
      ; "raptor_lake_avx2"
      ] )
  ; ( "radix25_z_n1t_avx2.c"
    , [ "25"; "--cil-n1t"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix25_z_n1t_bwd_avx2.c"
    , [ "25"; "--cil-n1t"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix25_z_t2_avx2.c"
    , [ "25"; "--cil-t2"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix25_z_t2_bwd_avx2.c"
    , [ "25"; "--cil-t2"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix25_z_t2_log3_avx2.c"
    , [ "25"; "--cil-t2"; "--cil-log3"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix25_z_t2_log3_bwd_avx2.c"
    , [ "25"
      ; "--cil-t2"
      ; "--cil-log3"
      ; "--cil-bwd"
      ; "--isa"
      ; "avx2"
      ; "--uarch"
      ; "raptor_lake_avx2"
      ] )
  ; ( "radix25_z_t2b_avx2.c"
    , [ "25"
      ; "--cil-t2"
      ; "--cil-blocked"
      ; "--cil-split"
      ; "5.5"
      ; "--isa"
      ; "avx2"
      ; "--uarch"
      ; "raptor_lake_avx2"
      ] )
  ; ( "radix25_z_t2b_bwd_avx2.c"
    , [ "25"
      ; "--cil-t2"
      ; "--cil-blocked"
      ; "--cil-split"
      ; "5.5"
      ; "--cil-bwd"
      ; "--isa"
      ; "avx2"
      ; "--uarch"
      ; "raptor_lake_avx2"
      ] )
  ; ( "radix25_z_t2b_log3_avx2.c"
    , [ "25"
      ; "--cil-t2"
      ; "--cil-blocked"
      ; "--cil-split"
      ; "5.5"
      ; "--cil-log3"
      ; "--isa"
      ; "avx2"
      ; "--uarch"
      ; "raptor_lake_avx2"
      ] )
  ; ( "radix25_z_t2b_log3_bwd_avx2.c"
    , [ "25"
      ; "--cil-t2"
      ; "--cil-blocked"
      ; "--cil-split"
      ; "5.5"
      ; "--cil-log3"
      ; "--cil-bwd"
      ; "--isa"
      ; "avx2"
      ; "--uarch"
      ; "raptor_lake_avx2"
      ] )
  ; ( "radix25_z_t2t_bwd_avx2.c"
    , [ "25"
      ; "--cil-t2"
      ; "--cil-turnst"
      ; "--cil-bwd"
      ; "--isa"
      ; "avx2"
      ; "--uarch"
      ; "raptor_lake_avx2"
      ] )
  ; ( "radix25_z_t2tg_bwd_avx2.c"
    , [ "25"
      ; "--cil-t2"
      ; "--cil-turnst-gs"
      ; "--cil-bwd"
      ; "--isa"
      ; "avx2"
      ; "--uarch"
      ; "raptor_lake_avx2"
      ] )
  ; ( "radix27_z_n1_avx2.c"
    , [ "27"; "--cil-n1"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix27_z_n1_bwd_avx2.c"
    , [ "27"; "--cil-n1"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix27_z_n1b_avx2.c"
    , [ "27"
      ; "--cil-n1"
      ; "--cil-blocked"
      ; "--cil-split"
      ; "3.9"
      ; "--isa"
      ; "avx2"
      ; "--uarch"
      ; "raptor_lake_avx2"
      ] )
  ; ( "radix27_z_n1b_bwd_avx2.c"
    , [ "27"
      ; "--cil-n1"
      ; "--cil-blocked"
      ; "--cil-split"
      ; "3.9"
      ; "--cil-bwd"
      ; "--isa"
      ; "avx2"
      ; "--uarch"
      ; "raptor_lake_avx2"
      ] )
  ; ( "radix27_z_n1t_avx2.c"
    , [ "27"; "--cil-n1t"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix27_z_n1t_bwd_avx2.c"
    , [ "27"; "--cil-n1t"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix27_z_t2_avx2.c"
    , [ "27"; "--cil-t2"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix27_z_t2_bwd_avx2.c"
    , [ "27"; "--cil-t2"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix27_z_t2_log3_avx2.c"
    , [ "27"; "--cil-t2"; "--cil-log3"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix27_z_t2_log3_bwd_avx2.c"
    , [ "27"
      ; "--cil-t2"
      ; "--cil-log3"
      ; "--cil-bwd"
      ; "--isa"
      ; "avx2"
      ; "--uarch"
      ; "raptor_lake_avx2"
      ] )
  ; ( "radix27_z_t2b_avx2.c"
    , [ "27"
      ; "--cil-t2"
      ; "--cil-blocked"
      ; "--cil-split"
      ; "3.9"
      ; "--isa"
      ; "avx2"
      ; "--uarch"
      ; "raptor_lake_avx2"
      ] )
  ; ( "radix27_z_t2b_bwd_avx2.c"
    , [ "27"
      ; "--cil-t2"
      ; "--cil-blocked"
      ; "--cil-split"
      ; "3.9"
      ; "--cil-bwd"
      ; "--isa"
      ; "avx2"
      ; "--uarch"
      ; "raptor_lake_avx2"
      ] )
  ; ( "radix27_z_t2b_log3_avx2.c"
    , [ "27"
      ; "--cil-t2"
      ; "--cil-blocked"
      ; "--cil-split"
      ; "3.9"
      ; "--cil-log3"
      ; "--isa"
      ; "avx2"
      ; "--uarch"
      ; "raptor_lake_avx2"
      ] )
  ; ( "radix27_z_t2b_log3_bwd_avx2.c"
    , [ "27"
      ; "--cil-t2"
      ; "--cil-blocked"
      ; "--cil-split"
      ; "3.9"
      ; "--cil-log3"
      ; "--cil-bwd"
      ; "--isa"
      ; "avx2"
      ; "--uarch"
      ; "raptor_lake_avx2"
      ] )
  ; ( "radix27_z_t2t_bwd_avx2.c"
    , [ "27"
      ; "--cil-t2"
      ; "--cil-turnst"
      ; "--cil-bwd"
      ; "--isa"
      ; "avx2"
      ; "--uarch"
      ; "raptor_lake_avx2"
      ] )
  ; ( "radix27_z_t2tg_bwd_avx2.c"
    , [ "27"
      ; "--cil-t2"
      ; "--cil-turnst-gs"
      ; "--cil-bwd"
      ; "--isa"
      ; "avx2"
      ; "--uarch"
      ; "raptor_lake_avx2"
      ] )
  ; ( "radix32_z_n1_bwd_avx2.c"
    , [ "32"; "--cil-n1"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix32_z_n1t_avx2.c"
    , [ "32"; "--cil-n1t"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix32_z_n1t_bwd_avx2.c"
    , [ "32"; "--cil-n1t"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix32_z_t2_avx2.c"
    , [ "32"; "--cil-t2"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix32_z_t2_bwd_avx2.c"
    , [ "32"; "--cil-t2"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix32_z_t2t_bwd_avx2.c"
    , [ "32"
      ; "--cil-t2"
      ; "--cil-turnst"
      ; "--cil-bwd"
      ; "--isa"
      ; "avx2"
      ; "--uarch"
      ; "raptor_lake_avx2"
      ] )
  ; ( "radix32_z_t2tg_bwd_avx2.c"
    , [ "32"
      ; "--cil-t2"
      ; "--cil-turnst-gs"
      ; "--cil-bwd"
      ; "--isa"
      ; "avx2"
      ; "--uarch"
      ; "raptor_lake_avx2"
      ] )
  ; ( "radix3_z_n1_avx2.c"
    , [ "3"; "--cil-n1"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix3_z_n1_bwd_avx2.c"
    , [ "3"; "--cil-n1"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix3_z_n1t_avx2.c"
    , [ "3"; "--cil-n1t"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix3_z_n1t_bwd_avx2.c"
    , [ "3"; "--cil-n1t"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix3_z_t2_avx2.c"
    , [ "3"; "--cil-t2"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix3_z_t2_bwd_avx2.c"
    , [ "3"; "--cil-t2"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix3_z_t2_log3_avx2.c"
    , [ "3"; "--cil-t2"; "--cil-log3"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix3_z_t2_log3_bwd_avx2.c"
    , [ "3"
      ; "--cil-t2"
      ; "--cil-log3"
      ; "--cil-bwd"
      ; "--isa"
      ; "avx2"
      ; "--uarch"
      ; "raptor_lake_avx2"
      ] )
  ; ( "radix3_z_t2t_bwd_avx2.c"
    , [ "3"
      ; "--cil-t2"
      ; "--cil-turnst"
      ; "--cil-bwd"
      ; "--isa"
      ; "avx2"
      ; "--uarch"
      ; "raptor_lake_avx2"
      ] )
  ; ( "radix3_z_t2tg_bwd_avx2.c"
    , [ "3"
      ; "--cil-t2"
      ; "--cil-turnst-gs"
      ; "--cil-bwd"
      ; "--isa"
      ; "avx2"
      ; "--uarch"
      ; "raptor_lake_avx2"
      ] )
  ; ( "radix45_z_n1b_avx2.c"
    , [ "45"
      ; "--cil-n1"
      ; "--cil-blocked"
      ; "--cil-split"
      ; "5.9"
      ; "--isa"
      ; "avx2"
      ; "--uarch"
      ; "raptor_lake_avx2"
      ] )
  ; ( "radix45_z_n1b_bwd_avx2.c"
    , [ "45"
      ; "--cil-n1"
      ; "--cil-blocked"
      ; "--cil-split"
      ; "5.9"
      ; "--cil-bwd"
      ; "--isa"
      ; "avx2"
      ; "--uarch"
      ; "raptor_lake_avx2"
      ] )
  ; ( "radix45_z_t2b_avx2.c"
    , [ "45"
      ; "--cil-t2"
      ; "--cil-blocked"
      ; "--cil-split"
      ; "5.9"
      ; "--isa"
      ; "avx2"
      ; "--uarch"
      ; "raptor_lake_avx2"
      ] )
  ; ( "radix45_z_t2b_bwd_avx2.c"
    , [ "45"
      ; "--cil-t2"
      ; "--cil-blocked"
      ; "--cil-split"
      ; "5.9"
      ; "--cil-bwd"
      ; "--isa"
      ; "avx2"
      ; "--uarch"
      ; "raptor_lake_avx2"
      ] )
  ; ( "radix45_z_t2b_log3_avx2.c"
    , [ "45"
      ; "--cil-t2"
      ; "--cil-blocked"
      ; "--cil-split"
      ; "5.9"
      ; "--cil-log3"
      ; "--isa"
      ; "avx2"
      ; "--uarch"
      ; "raptor_lake_avx2"
      ] )
  ; ( "radix45_z_t2b_log3_bwd_avx2.c"
    , [ "45"
      ; "--cil-t2"
      ; "--cil-blocked"
      ; "--cil-split"
      ; "5.9"
      ; "--cil-log3"
      ; "--cil-bwd"
      ; "--isa"
      ; "avx2"
      ; "--uarch"
      ; "raptor_lake_avx2"
      ] )
  ; ( "radix49_z_n1b_avx2.c"
    , [ "49"
      ; "--cil-n1"
      ; "--cil-blocked"
      ; "--cil-split"
      ; "7.7"
      ; "--isa"
      ; "avx2"
      ; "--uarch"
      ; "raptor_lake_avx2"
      ] )
  ; ( "radix49_z_n1b_bwd_avx2.c"
    , [ "49"
      ; "--cil-n1"
      ; "--cil-blocked"
      ; "--cil-split"
      ; "7.7"
      ; "--cil-bwd"
      ; "--isa"
      ; "avx2"
      ; "--uarch"
      ; "raptor_lake_avx2"
      ] )
  ; ( "radix49_z_t2b_avx2.c"
    , [ "49"
      ; "--cil-t2"
      ; "--cil-blocked"
      ; "--cil-split"
      ; "7.7"
      ; "--isa"
      ; "avx2"
      ; "--uarch"
      ; "raptor_lake_avx2"
      ] )
  ; ( "radix49_z_t2b_bwd_avx2.c"
    , [ "49"
      ; "--cil-t2"
      ; "--cil-blocked"
      ; "--cil-split"
      ; "7.7"
      ; "--cil-bwd"
      ; "--isa"
      ; "avx2"
      ; "--uarch"
      ; "raptor_lake_avx2"
      ] )
  ; ( "radix49_z_t2b_log3_avx2.c"
    , [ "49"
      ; "--cil-t2"
      ; "--cil-blocked"
      ; "--cil-split"
      ; "7.7"
      ; "--cil-log3"
      ; "--isa"
      ; "avx2"
      ; "--uarch"
      ; "raptor_lake_avx2"
      ] )
  ; ( "radix49_z_t2b_log3_bwd_avx2.c"
    , [ "49"
      ; "--cil-t2"
      ; "--cil-blocked"
      ; "--cil-split"
      ; "7.7"
      ; "--cil-log3"
      ; "--cil-bwd"
      ; "--isa"
      ; "avx2"
      ; "--uarch"
      ; "raptor_lake_avx2"
      ] )
  ; ( "radix4_z_n1_bwd_avx2.c"
    , [ "4"; "--cil-n1"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix4_z_n1t_avx2.c"
    , [ "4"; "--cil-n1t"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix4_z_n1t_bwd_avx2.c"
    , [ "4"; "--cil-n1t"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix4_z_t2_avx2.c"
    , [ "4"; "--cil-t2"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix4_z_t2_bwd_avx2.c"
    , [ "4"; "--cil-t2"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix4_z_t2t_bwd_avx2.c"
    , [ "4"
      ; "--cil-t2"
      ; "--cil-turnst"
      ; "--cil-bwd"
      ; "--isa"
      ; "avx2"
      ; "--uarch"
      ; "raptor_lake_avx2"
      ] )
  ; ( "radix4_z_t2tg_bwd_avx2.c"
    , [ "4"
      ; "--cil-t2"
      ; "--cil-turnst-gs"
      ; "--cil-bwd"
      ; "--isa"
      ; "avx2"
      ; "--uarch"
      ; "raptor_lake_avx2"
      ] )
  ; ( "radix5_z_n1_avx2.c"
    , [ "5"; "--cil-n1"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix5_z_n1_bwd_avx2.c"
    , [ "5"; "--cil-n1"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix5_z_n1t_avx2.c"
    , [ "5"; "--cil-n1t"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix5_z_n1t_bwd_avx2.c"
    , [ "5"; "--cil-n1t"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix5_z_t2_avx2.c"
    , [ "5"; "--cil-t2"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix5_z_t2_bwd_avx2.c"
    , [ "5"; "--cil-t2"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix5_z_t2_log3_avx2.c"
    , [ "5"; "--cil-t2"; "--cil-log3"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix5_z_t2_log3_bwd_avx2.c"
    , [ "5"
      ; "--cil-t2"
      ; "--cil-log3"
      ; "--cil-bwd"
      ; "--isa"
      ; "avx2"
      ; "--uarch"
      ; "raptor_lake_avx2"
      ] )
  ; ( "radix5_z_t2t_bwd_avx2.c"
    , [ "5"
      ; "--cil-t2"
      ; "--cil-turnst"
      ; "--cil-bwd"
      ; "--isa"
      ; "avx2"
      ; "--uarch"
      ; "raptor_lake_avx2"
      ] )
  ; ( "radix5_z_t2tg_bwd_avx2.c"
    , [ "5"
      ; "--cil-t2"
      ; "--cil-turnst-gs"
      ; "--cil-bwd"
      ; "--isa"
      ; "avx2"
      ; "--uarch"
      ; "raptor_lake_avx2"
      ] )
  ; ( "radix64_z_n1_bwd_avx2.c"
    , [ "64"; "--cil-n1"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix64_z_n1t_avx2.c"
    , [ "64"; "--cil-n1t"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix64_z_n1t_bwd_avx2.c"
    , [ "64"; "--cil-n1t"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix64_z_t2_avx2.c"
    , [ "64"; "--cil-t2"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix64_z_t2_bwd_avx2.c"
    , [ "64"; "--cil-t2"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix64_z_t2t_bwd_avx2.c"
    , [ "64"
      ; "--cil-t2"
      ; "--cil-turnst"
      ; "--cil-bwd"
      ; "--isa"
      ; "avx2"
      ; "--uarch"
      ; "raptor_lake_avx2"
      ] )
  ; ( "radix64_z_t2tg_bwd_avx2.c"
    , [ "64"
      ; "--cil-t2"
      ; "--cil-turnst-gs"
      ; "--cil-bwd"
      ; "--isa"
      ; "avx2"
      ; "--uarch"
      ; "raptor_lake_avx2"
      ] )
  ; ( "radix6_z_n1_avx2.c"
    , [ "6"; "--cil-n1"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix6_z_n1_bwd_avx2.c"
    , [ "6"; "--cil-n1"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix6_z_n1t_avx2.c"
    , [ "6"; "--cil-n1t"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix6_z_n1t_bwd_avx2.c"
    , [ "6"; "--cil-n1t"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix6_z_t2_avx2.c"
    , [ "6"; "--cil-t2"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix6_z_t2_bwd_avx2.c"
    , [ "6"; "--cil-t2"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix6_z_t2t_bwd_avx2.c"
    , [ "6"
      ; "--cil-t2"
      ; "--cil-turnst"
      ; "--cil-bwd"
      ; "--isa"
      ; "avx2"
      ; "--uarch"
      ; "raptor_lake_avx2"
      ] )
  ; ( "radix6_z_t2tg_bwd_avx2.c"
    , [ "6"
      ; "--cil-t2"
      ; "--cil-turnst-gs"
      ; "--cil-bwd"
      ; "--isa"
      ; "avx2"
      ; "--uarch"
      ; "raptor_lake_avx2"
      ] )
  ; ( "radix7_z_n1_avx2.c"
    , [ "7"; "--cil-n1"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix7_z_n1_bwd_avx2.c"
    , [ "7"; "--cil-n1"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix7_z_n1t_avx2.c"
    , [ "7"; "--cil-n1t"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix7_z_n1t_bwd_avx2.c"
    , [ "7"; "--cil-n1t"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix7_z_t2_avx2.c"
    , [ "7"; "--cil-t2"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix7_z_t2_bwd_avx2.c"
    , [ "7"; "--cil-t2"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix7_z_t2_log3_avx2.c"
    , [ "7"; "--cil-t2"; "--cil-log3"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix7_z_t2_log3_bwd_avx2.c"
    , [ "7"
      ; "--cil-t2"
      ; "--cil-log3"
      ; "--cil-bwd"
      ; "--isa"
      ; "avx2"
      ; "--uarch"
      ; "raptor_lake_avx2"
      ] )
  ; ( "radix7_z_t2t_bwd_avx2.c"
    , [ "7"
      ; "--cil-t2"
      ; "--cil-turnst"
      ; "--cil-bwd"
      ; "--isa"
      ; "avx2"
      ; "--uarch"
      ; "raptor_lake_avx2"
      ] )
  ; ( "radix7_z_t2tg_bwd_avx2.c"
    , [ "7"
      ; "--cil-t2"
      ; "--cil-turnst-gs"
      ; "--cil-bwd"
      ; "--isa"
      ; "avx2"
      ; "--uarch"
      ; "raptor_lake_avx2"
      ] )
  ; ( "radix8_z_n1_bwd_avx2.c"
    , [ "8"; "--cil-n1"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix8_z_n1t_avx2.c"
    , [ "8"; "--cil-n1t"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix8_z_n1t_bwd_avx2.c"
    , [ "8"; "--cil-n1t"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix8_z_t2_avx2.c"
    , [ "8"; "--cil-t2"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix8_z_t2_bwd_avx2.c"
    , [ "8"; "--cil-t2"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix8_z_t2t_bwd_avx2.c"
    , [ "8"
      ; "--cil-t2"
      ; "--cil-turnst"
      ; "--cil-bwd"
      ; "--isa"
      ; "avx2"
      ; "--uarch"
      ; "raptor_lake_avx2"
      ] )
  ; ( "radix8_z_t2tg_bwd_avx2.c"
    , [ "8"
      ; "--cil-t2"
      ; "--cil-turnst-gs"
      ; "--cil-bwd"
      ; "--isa"
      ; "avx2"
      ; "--uarch"
      ; "raptor_lake_avx2"
      ] )
  ; ( "radix9_z_n1_avx2.c"
    , [ "9"; "--cil-n1"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix9_z_n1_bwd_avx2.c"
    , [ "9"; "--cil-n1"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix9_z_n1b_avx2.c"
    , [ "9"
      ; "--cil-n1"
      ; "--cil-blocked"
      ; "--cil-split"
      ; "3.3"
      ; "--isa"
      ; "avx2"
      ; "--uarch"
      ; "raptor_lake_avx2"
      ] )
  ; ( "radix9_z_n1b_bwd_avx2.c"
    , [ "9"
      ; "--cil-n1"
      ; "--cil-blocked"
      ; "--cil-split"
      ; "3.3"
      ; "--cil-bwd"
      ; "--isa"
      ; "avx2"
      ; "--uarch"
      ; "raptor_lake_avx2"
      ] )
  ; ( "radix9_z_n1t_avx2.c"
    , [ "9"; "--cil-n1t"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix9_z_n1t_bwd_avx2.c"
    , [ "9"; "--cil-n1t"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix9_z_t2_avx2.c"
    , [ "9"; "--cil-t2"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix9_z_t2_bwd_avx2.c"
    , [ "9"; "--cil-t2"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix9_z_t2_log3_avx2.c"
    , [ "9"; "--cil-t2"; "--cil-log3"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix9_z_t2_log3_bwd_avx2.c"
    , [ "9"
      ; "--cil-t2"
      ; "--cil-log3"
      ; "--cil-bwd"
      ; "--isa"
      ; "avx2"
      ; "--uarch"
      ; "raptor_lake_avx2"
      ] )
  ; ( "radix9_z_t2b_avx2.c"
    , [ "9"
      ; "--cil-t2"
      ; "--cil-blocked"
      ; "--cil-split"
      ; "3.3"
      ; "--isa"
      ; "avx2"
      ; "--uarch"
      ; "raptor_lake_avx2"
      ] )
  ; ( "radix9_z_t2b_bwd_avx2.c"
    , [ "9"
      ; "--cil-t2"
      ; "--cil-blocked"
      ; "--cil-split"
      ; "3.3"
      ; "--cil-bwd"
      ; "--isa"
      ; "avx2"
      ; "--uarch"
      ; "raptor_lake_avx2"
      ] )
  ; ( "radix9_z_t2b_log3_avx2.c"
    , [ "9"
      ; "--cil-t2"
      ; "--cil-blocked"
      ; "--cil-split"
      ; "3.3"
      ; "--cil-log3"
      ; "--isa"
      ; "avx2"
      ; "--uarch"
      ; "raptor_lake_avx2"
      ] )
  ; ( "radix9_z_t2b_log3_bwd_avx2.c"
    , [ "9"
      ; "--cil-t2"
      ; "--cil-blocked"
      ; "--cil-split"
      ; "3.3"
      ; "--cil-log3"
      ; "--cil-bwd"
      ; "--isa"
      ; "avx2"
      ; "--uarch"
      ; "raptor_lake_avx2"
      ] )
  ; ( "radix9_z_t2t_bwd_avx2.c"
    , [ "9"
      ; "--cil-t2"
      ; "--cil-turnst"
      ; "--cil-bwd"
      ; "--isa"
      ; "avx2"
      ; "--uarch"
      ; "raptor_lake_avx2"
      ] )
  ; ( "radix9_z_t2tg_bwd_avx2.c"
    , [ "9"
      ; "--cil-t2"
      ; "--cil-turnst-gs"
      ; "--cil-bwd"
      ; "--isa"
      ; "avx2"
      ; "--uarch"
      ; "raptor_lake_avx2"
      ] )
  ]
;;

(* ── M10b: THE COVERAGE RAISE — the 75 reproducing replay/k1 cells ──
   Four REGISTRY-INVISIBLE quadrants (no registry emitter names them, so
   the registries stay byte-identical — these families dispatch via
   wisdom/il2p, never via registry slots): the zr2c strided-r2c pairs,
   and the oop edge/variant cells (post-tw · UL/UGUL edges · spec
   extras) plus the six k1_mono champions.  LITERAL cells from recorded
   provenance, same law and entry-gate treatment as the M12a zil
   quadrants (75/75 byte-identical at entry).  Excluded, with cause:
   the non-reproducing 29 (orphans / tangent / stale — pool-sunset or
   announced-numeric-event material); the 9 sed-renamed zil variants
   (need an emitter suffix knob); and radix256_r2c_term_ls_r8_avx512 —
   its recorded argv carries --isa AFTER --emit-c (the M5 placement
   quirk), which emit_one's append-last convention cannot reproduce;
   it enters when that family's argv order is normalized (naturally
   alongside its known ULP defect, codelet_corpus_reproducibility). *)
let oop_edges_avx2_cells : (string * string list) list =
    [ "radix10_t1_dif_oop_avx2.c", [ "10"; "--twiddled"; "--post-tw"; "--oop"; "--oop-buffer-oop"; "--oop-load"; "UG"; "--oop-store"; "UG"; "--isa"; "avx2" ]
    ; "radix16_n1_oop_ugul_avx2.c", [ "16"; "--oop"; "--oop-buffer-oop"; "--oop-load"; "UG"; "--oop-store"; "UL"; "--isa"; "avx2" ]
    ; "radix16_t1_oop_ugug_log3_avx2.c", [ "16"; "--oop"; "--oop-buffer-oop"; "--oop-load"; "UG"; "--oop-store"; "UG"; "--isa"; "avx2"; "--twiddled"; "--log3" ]
    ; "radix16_t1_oop_ul_avx2.c", [ "16"; "--oop"; "--oop-buffer-oop"; "--oop-load"; "UL"; "--oop-store"; "UG"; "--isa"; "avx2"; "--twiddled" ]
    ; "radix16_t1_oop_ul_log3_avx2.c", [ "16"; "--oop"; "--oop-buffer-oop"; "--oop-load"; "UL"; "--oop-store"; "UG"; "--isa"; "avx2"; "--twiddled"; "--log3" ]
    ; "radix16_t1_oop_ul_twl_avx2.c", [ "16"; "--oop"; "--oop-buffer-oop"; "--oop-load"; "UL"; "--oop-store"; "UG"; "--isa"; "avx2"; "--twiddled"; "--oop-tw-linear" ]
    ; "radix20_t1_dif_oop_avx2.c", [ "20"; "--twiddled"; "--post-tw"; "--oop"; "--oop-buffer-oop"; "--oop-load"; "UG"; "--oop-store"; "UG"; "--isa"; "avx2" ]
    ; "radix25_t1_dif_oop_avx2.c", [ "25"; "--twiddled"; "--post-tw"; "--oop"; "--oop-buffer-oop"; "--oop-load"; "UG"; "--oop-store"; "UG"; "--isa"; "avx2" ]
    ; "radix32_n1_oop_ugul_avx2.c", [ "32"; "--oop"; "--oop-buffer-oop"; "--oop-load"; "UG"; "--oop-store"; "UL"; "--isa"; "avx2" ]
    ; "radix32_n1_oop_ugul_spec1024_avx2.c", [ "32"; "--oop"; "--oop-buffer-oop"; "--oop-load"; "UG"; "--oop-store"; "UL"; "--isa"; "avx2"; "--oop-strides"; "32,1,1,32"; "--oop-spec-named" ]
    ; "radix32_t1_oop_spec1024_avx2.c", [ "32"; "--oop"; "--oop-buffer-oop"; "--oop-load"; "UG"; "--oop-store"; "UG"; "--isa"; "avx2"; "--twiddled"; "--oop-strides"; "32,1,32,1"; "--oop-spec-named" ]
    ; "radix32_t1_oop_ugug_log3_avx2.c", [ "32"; "--oop"; "--oop-buffer-oop"; "--oop-load"; "UG"; "--oop-store"; "UG"; "--isa"; "avx2"; "--twiddled"; "--log3" ]
    ; "radix32_t1_oop_ul_avx2.c", [ "32"; "--oop"; "--oop-buffer-oop"; "--oop-load"; "UL"; "--oop-store"; "UG"; "--isa"; "avx2"; "--twiddled" ]
    ; "radix32_t1_oop_ul_log3_avx2.c", [ "32"; "--oop"; "--oop-buffer-oop"; "--oop-load"; "UL"; "--oop-store"; "UG"; "--isa"; "avx2"; "--twiddled"; "--log3" ]
    ; "radix32_t1_oop_ul_twl_avx2.c", [ "32"; "--oop"; "--oop-buffer-oop"; "--oop-load"; "UL"; "--oop-store"; "UG"; "--isa"; "avx2"; "--twiddled"; "--oop-tw-linear" ]
    ; "radix4_n1_oop_ugul_avx2.c", [ "4"; "--oop"; "--oop-buffer-oop"; "--oop-load"; "UG"; "--oop-store"; "UL"; "--isa"; "avx2" ]
    ; "radix4_t1_oop_ugug_log3_avx2.c", [ "4"; "--oop"; "--oop-buffer-oop"; "--oop-load"; "UG"; "--oop-store"; "UG"; "--isa"; "avx2"; "--twiddled"; "--log3" ]
    ; "radix4_t1_oop_ul_avx2.c", [ "4"; "--oop"; "--oop-buffer-oop"; "--oop-load"; "UL"; "--oop-store"; "UG"; "--isa"; "avx2"; "--twiddled" ]
    ; "radix4_t1_oop_ul_log3_avx2.c", [ "4"; "--oop"; "--oop-buffer-oop"; "--oop-load"; "UL"; "--oop-store"; "UG"; "--isa"; "avx2"; "--twiddled"; "--log3" ]
    ; "radix4_t1_oop_ul_twl_avx2.c", [ "4"; "--oop"; "--oop-buffer-oop"; "--oop-load"; "UL"; "--oop-store"; "UG"; "--isa"; "avx2"; "--twiddled"; "--oop-tw-linear" ]
    ; "radix5_t1_dif_oop_avx2.c", [ "5"; "--twiddled"; "--post-tw"; "--oop"; "--oop-buffer-oop"; "--oop-load"; "UG"; "--oop-store"; "UG"; "--isa"; "avx2" ]
    ; "radix64_n1_oop_spec4096_avx2.c", [ "64"; "--oop"; "--oop-buffer-oop"; "--oop-load"; "UG"; "--oop-store"; "UG"; "--isa"; "avx2"; "--oop-strides"; "64,1,64,1"; "--oop-spec-named" ]
    ; "radix64_n1_oop_ugul_avx2.c", [ "64"; "--oop"; "--oop-buffer-oop"; "--oop-load"; "UG"; "--oop-store"; "UL"; "--isa"; "avx2" ]
    ; "radix64_t1_oop_ugug_log3_avx2.c", [ "64"; "--oop"; "--oop-buffer-oop"; "--oop-load"; "UG"; "--oop-store"; "UG"; "--isa"; "avx2"; "--twiddled"; "--log3" ]
    ; "radix64_t1_oop_ul_avx2.c", [ "64"; "--oop"; "--oop-buffer-oop"; "--oop-load"; "UL"; "--oop-store"; "UG"; "--isa"; "avx2"; "--twiddled" ]
    ; "radix64_t1_oop_ul_log3_avx2.c", [ "64"; "--oop"; "--oop-buffer-oop"; "--oop-load"; "UL"; "--oop-store"; "UG"; "--isa"; "avx2"; "--twiddled"; "--log3" ]
    ; "radix64_t1_oop_ul_log3_spec4096_avx2.c", [ "64"; "--oop"; "--oop-buffer-oop"; "--oop-load"; "UL"; "--oop-store"; "UG"; "--isa"; "avx2"; "--twiddled"; "--log3"; "--oop-strides"; "1,64,64,1"; "--oop-spec-named" ]
    ; "radix64_t1_oop_ul_twl_avx2.c", [ "64"; "--oop"; "--oop-buffer-oop"; "--oop-load"; "UL"; "--oop-store"; "UG"; "--isa"; "avx2"; "--twiddled"; "--oop-tw-linear" ]
    ; "radix8_n1_oop_ugul_avx2.c", [ "8"; "--oop"; "--oop-buffer-oop"; "--oop-load"; "UG"; "--oop-store"; "UL"; "--isa"; "avx2" ]
    ; "radix8_t1_oop_ugug_log3_avx2.c", [ "8"; "--oop"; "--oop-buffer-oop"; "--oop-load"; "UG"; "--oop-store"; "UG"; "--isa"; "avx2"; "--twiddled"; "--log3" ]
    ; "radix8_t1_oop_ul_avx2.c", [ "8"; "--oop"; "--oop-buffer-oop"; "--oop-load"; "UL"; "--oop-store"; "UG"; "--isa"; "avx2"; "--twiddled" ]
    ; "radix8_t1_oop_ul_log3_avx2.c", [ "8"; "--oop"; "--oop-buffer-oop"; "--oop-load"; "UL"; "--oop-store"; "UG"; "--isa"; "avx2"; "--twiddled"; "--log3" ]
    ; "radix8_t1_oop_ul_twl_avx2.c", [ "8"; "--oop"; "--oop-buffer-oop"; "--oop-load"; "UL"; "--oop-store"; "UG"; "--isa"; "avx2"; "--twiddled"; "--oop-tw-linear" ]
    ; "vfft_k1_mono128_16x8_avx2.c", [ "128"; "--k1-mono"; "--k1-r1"; "16"; "--isa"; "avx2" ]
    ; "vfft_k1_mono128_8x16_avx2.c", [ "128"; "--k1-mono"; "--k1-r1"; "8"; "--isa"; "avx2" ]
    ; "vfft_k1_mono256_16x16_avx2.c", [ "256"; "--k1-mono"; "--k1-r1"; "16"; "--isa"; "avx2" ]
    ; "vfft_k1_mono64_avx2.c", [ "64"; "--k1-mono"; "--isa"; "avx2" ]
    ; "vfft_k1_mono64_il_bwd_avx2.c", [ "64"; "--k1-mono"; "--k1-il"; "--k1-sw"; "--isa"; "avx2" ]
    ; "vfft_k1_mono64_il_fwd_avx2.c", [ "64"; "--k1-mono"; "--k1-il"; "--isa"; "avx2" ]
    ]
;;

let oop_edges_avx512_cells : (string * string list) list =
    [ "radix10_t1_dif_oop_avx512.c", [ "10"; "--twiddled"; "--post-tw"; "--oop"; "--oop-buffer-oop"; "--oop-load"; "UG"; "--oop-store"; "UG"; "--isa"; "avx512" ]
    ; "radix20_t1_dif_oop_avx512.c", [ "20"; "--twiddled"; "--post-tw"; "--oop"; "--oop-buffer-oop"; "--oop-load"; "UG"; "--oop-store"; "UG"; "--isa"; "avx512" ]
    ; "radix25_t1_dif_oop_avx512.c", [ "25"; "--twiddled"; "--post-tw"; "--oop"; "--oop-buffer-oop"; "--oop-load"; "UG"; "--oop-store"; "UG"; "--isa"; "avx512" ]
    ; "radix5_t1_dif_oop_avx512.c", [ "5"; "--twiddled"; "--post-tw"; "--oop"; "--oop-buffer-oop"; "--oop-load"; "UG"; "--oop-store"; "UG"; "--isa"; "avx512" ]
    ]
;;

let strided_r2c_avx2_cells : (string * string list) list =
    [ "r128_n1_bwd_strided_r2c.c", [ "128"; "--strided-r2c"; "--bwd"; "--isa"; "avx2" ]
    ; "r128_n1_fwd_strided_r2c.c", [ "128"; "--strided-r2c"; "--isa"; "avx2" ]
    ; "r12_n1_bwd_strided_r2c.c", [ "12"; "--strided-r2c"; "--bwd"; "--isa"; "avx2" ]
    ; "r12_n1_fwd_strided_r2c.c", [ "12"; "--strided-r2c"; "--isa"; "avx2" ]
    ; "r16_n1_bwd_strided_r2c.c", [ "16"; "--strided-r2c"; "--bwd"; "--isa"; "avx2" ]
    ; "r16_n1_fwd_strided_r2c.c", [ "16"; "--strided-r2c"; "--isa"; "avx2" ]
    ; "r20_n1_bwd_strided_r2c.c", [ "20"; "--strided-r2c"; "--bwd"; "--isa"; "avx2" ]
    ; "r20_n1_fwd_strided_r2c.c", [ "20"; "--strided-r2c"; "--isa"; "avx2" ]
    ; "r256_n1_bwd_strided_r2c.c", [ "256"; "--strided-r2c"; "--bwd"; "--isa"; "avx2" ]
    ; "r256_n1_fwd_strided_r2c.c", [ "256"; "--strided-r2c"; "--isa"; "avx2" ]
    ; "r32_n1_bwd_strided_r2c.c", [ "32"; "--strided-r2c"; "--bwd"; "--isa"; "avx2" ]
    ; "r32_n1_fwd_strided_r2c.c", [ "32"; "--strided-r2c"; "--isa"; "avx2" ]
    ; "r512_n1_bwd_strided_r2c.c", [ "512"; "--strided-r2c"; "--bwd"; "--isa"; "avx2" ]
    ; "r512_n1_fwd_strided_r2c.c", [ "512"; "--strided-r2c"; "--isa"; "avx2" ]
    ; "r64_n1_bwd_strided_r2c.c", [ "64"; "--strided-r2c"; "--bwd"; "--isa"; "avx2" ]
    ; "r64_n1_fwd_strided_r2c.c", [ "64"; "--strided-r2c"; "--isa"; "avx2" ]
    ; "r8_n1_bwd_strided_r2c.c", [ "8"; "--strided-r2c"; "--bwd"; "--isa"; "avx2" ]
    ; "r8_n1_fwd_strided_r2c.c", [ "8"; "--strided-r2c"; "--isa"; "avx2" ]
    ]
;;

let strided_r2c_avx512_cells : (string * string list) list =
    [ "r128_n1_bwd_strided_r2c.c", [ "128"; "--strided-r2c"; "--bwd"; "--isa"; "avx512" ]
    ; "r128_n1_fwd_strided_r2c.c", [ "128"; "--strided-r2c"; "--isa"; "avx512" ]
    ; "r16_n1_bwd_strided_r2c.c", [ "16"; "--strided-r2c"; "--bwd"; "--isa"; "avx512" ]
    ; "r16_n1_fwd_strided_r2c.c", [ "16"; "--strided-r2c"; "--isa"; "avx512" ]
    ; "r256_n1_bwd_strided_r2c.c", [ "256"; "--strided-r2c"; "--bwd"; "--isa"; "avx512" ]
    ; "r256_n1_fwd_strided_r2c.c", [ "256"; "--strided-r2c"; "--isa"; "avx512" ]
    ; "r32_n1_bwd_strided_r2c.c", [ "32"; "--strided-r2c"; "--bwd"; "--isa"; "avx512" ]
    ; "r32_n1_fwd_strided_r2c.c", [ "32"; "--strided-r2c"; "--isa"; "avx512" ]
    ; "r512_n1_bwd_strided_r2c.c", [ "512"; "--strided-r2c"; "--bwd"; "--isa"; "avx512" ]
    ; "r512_n1_fwd_strided_r2c.c", [ "512"; "--strided-r2c"; "--isa"; "avx512" ]
    ; "r64_n1_bwd_strided_r2c.c", [ "64"; "--strided-r2c"; "--bwd"; "--isa"; "avx512" ]
    ; "r64_n1_fwd_strided_r2c.c", [ "64"; "--strided-r2c"; "--isa"; "avx512" ]
    ; "r8_n1_bwd_strided_r2c.c", [ "8"; "--strided-r2c"; "--bwd"; "--isa"; "avx512" ]
    ; "r8_n1_fwd_strided_r2c.c", [ "8"; "--strided-r2c"; "--isa"; "avx512" ]
    ]
;;

let matrix_files (quadrant : string) : (string * string list) list =
  match quadrant with
  | "oop-edges-avx2" -> oop_edges_avx2_cells
  | "oop-edges-avx512" -> oop_edges_avx512_cells
  | "strided-r2c-avx2" -> strided_r2c_avx2_cells
  | "strided-r2c-avx512" -> strided_r2c_avx512_cells
  | "zil-boundary" -> zil_boundary_cells
  | "zil-pure" -> zil_pure_cells
  | "inplace-avx2" | "inplace-avx512" ->
    let isa = if quadrant = "inplace-avx2" then "avx2" else "avx512" in
    List.concat_map
      (fun r ->
         List.map
           (fun (fam, args) -> Printf.sprintf "r%d_%s.c" r fam, string_of_int r :: args)
           (ip_families isa))
      ip_radices
  | "oop-avx2" | "oop-avx512" ->
    let isa = if quadrant = "oop-avx2" then "avx2" else "avx512" in
    let base = oop_base isa in
    let n1 =
      List.map
        (fun r -> Printf.sprintf "radix%d_n1_oop_%s.c" r isa, string_of_int r :: base)
        oop_n1_radices
    in
    let t1p =
      List.concat_map
        (fun r ->
           [ ( Printf.sprintf "radix%d_t1p_oop_%s.c" r isa
             , (string_of_int r :: base) @ [ "--twiddled-pos" ] )
           ; ( Printf.sprintf "radix%d_t1p_log3_oop_%s.c" r isa
             , (string_of_int r :: base) @ [ "--twiddled-pos"; "--log3" ] )
           ])
        t1p_radices
    in
    (* PER-GROUP (per-lane) twiddle second-stage codelet (--twiddled, NOT
     * --twiddled-pos). Twiddle is loadu(tw[j*me+b]) per group, so it's
     * arbitrary-K-correct (no k2-boundary straddle) and rem-aware-maskable —
     * the BAILEY2 s2 codelet for odd K (docs arbitrary_k). Forward only; OOP
     * backward is a pointer-swap on the forward plan. *)
    let t1 =
      List.map
        (fun r ->
           ( Printf.sprintf "radix%d_t1_oop_%s.c" r isa
           , (string_of_int r :: base) @ [ "--twiddled" ] ))
        t1p_radices
    in
    let extras =
      (* OOP stride-specialized codelets (--oop-strides): strides baked as
       * compile-time constants -> 7-arg ABI, ~6-10% over runtime-stride
       * (doc oop_stride_specialization.md). The lane-blocked geometry is
       * rv = R*V where V is the ISA vector width (avx512 V=8, avx2 V=4),
       * verified against the emitted codelets (b += 8 / b += 4). Earlier this
       * was avx512-only with rv hardcoded r*8; section 65 made it lane-aware
       * and extended it to avx2 with rv = r*4. (t1s_oop was removed in
       * section 64: --t1s without --twiddled is a no-op n1 duplicate.) *)
      let v = if isa = "avx512" then 8 else 4 in
      let opt = [ "--fuse"; string_of_int v; "--oop-store-fused" ] in
      List.concat_map
        (fun r ->
           let rv = r * v in
           let strides_n1 = Printf.sprintf "%d,1,%d,%d" rv v r in
           let strides_t1p = Printf.sprintf "%d,1,%d,1" rv rv in
           [ ( Printf.sprintf "radix%d_n1_oop_%s_spec.c" r isa
             , (string_of_int r :: base) @ opt @ [ "--oop-strides"; strides_n1 ] )
           ; ( Printf.sprintf "radix%d_t1p_oop_%s_spec.c" r isa
             , (string_of_int r :: base)
               @ opt
               @ [ "--oop-strides"; strides_t1p; "--twiddled-pos" ] )
           ; ( Printf.sprintf "radix%d_t1p_log3_oop_%s_spec.c" r isa
             , (string_of_int r :: base)
               @ opt
               @ [ "--oop-strides"; strides_t1p; "--twiddled-pos"; "--log3" ] )
           ])
        spec_radices
    in
    n1 @ t1p @ t1 @ extras
  | "strided-avx2" | "strided-avx512" ->
    let isa = if quadrant = "strided-avx2" then "avx2" else "avx512" in
    let radices = if isa = "avx2" then strided_radices_avx2 else strided_radices_avx512 in
    List.concat_map
      (fun r ->
         [ ( Printf.sprintf "r%d_n1_fwd_strided.c" r
           , [ string_of_int r; "--strided"; "--isa"; isa ] )
         ; ( Printf.sprintf "r%d_n1_bwd_strided.c" r
           , [ string_of_int r; "--strided"; "--isa"; isa; "--bwd" ] )
         ])
      radices
  | "rfft-avx2" | "rfft-avx512" ->
    (* Native real-cascade family (sections 60-61,
     * docs/native_rfft_design.md): leaf + middle + terminator codelets
     * for the r2hc executor (P2). r2cf: stride_n1_fn ABI (7-arg, in_im
     * present but never read). hc2hc/hc2c: generic 7-arg with runtime
     * twiddles, slot 0 never loaded (NaN-poison proved, section 60).
     * Forward only in P1; backward (r2cb + bwd cascades) lands with the
     * c2r phase. All 28 codelets per ISA gated in sections 60-61. *)
    let isa = if quadrant = "rfft-avx2" then "avx2" else "avx512" in
    let radices = [ 2; 3; 4; 5; 7; 8; 16 ] in
    let leaf_only = [ 32 ] in
    (* big leaves for low-stage-count plans;
                                 no hc combine at these radices *)
    List.map
      (fun r ->
         ( Printf.sprintf "radix%d_r2cf_%s.c" r isa
         , [ string_of_int r; "--r2cf"; "--isa"; isa; "--su" ] ))
      leaf_only
    @ List.concat_map
        (fun r ->
           [ ( Printf.sprintf "radix%d_r2cf_%s.c" r isa
             , [ string_of_int r; "--r2cf"; "--isa"; isa; "--su" ] )
           ; ( Printf.sprintf "radix%d_hc2hc_dit_fwd_%s.c" r isa
             , [ string_of_int r; "--hc2hc"; "--t1s"; "--isa"; isa; "--su" ] )
           ; ( Printf.sprintf "radix%d_hc2hc_dif_fwd_%s.c" r isa
             , [ string_of_int r; "--hc2hc"; "--dif"; "--t1s"; "--isa"; isa; "--su" ] )
           ; ( Printf.sprintf "radix%d_hc2c_dit_fwd_%s.c" r isa
             , [ string_of_int r; "--hc2c"; "--t1s"; "--isa"; isa; "--su" ] )
           ; (* log3 variants (section 62: hc2cf2 = hc2c + log3). FFTW's hc2cf2
              * family is literally hc2c generated with -twiddle-log3; here the
              * --log3 flag composes with --t1s the same way (verified: 7->3
              * twiddle slots, op counts match FFTW hc2cf2_8 at 74 add/30 fma).
              * The log3 twiddle stage (hc2hc) and the log3 NATURAL terminator
              * (hc2c-nat, the 6-ptr mirror-pair ABI the rfft executor's stage-0
              * actually calls — NOT the packed 4-ptr hc2c). *)
             ( Printf.sprintf "radix%d_hc2hc_dit_log3_fwd_%s.c" r isa
             , [ string_of_int r; "--hc2hc"; "--log3"; "--t1s"; "--isa"; isa; "--su" ] )
           ; ( Printf.sprintf "radix%d_hc2c_nat_log3_fwd_%s.c" r isa
             , [ string_of_int r; "--hc2c-nat"; "--log3"; "--t1s"; "--isa"; isa; "--su" ]
             )
           ; ( Printf.sprintf "radix%d_hc2c_nat_fwd_%s.c" r isa
             , [ string_of_int r; "--hc2c-nat"; "--t1s"; "--isa"; isa; "--su" ] )
           ; ( Printf.sprintf "radix%d_hc2hc_dit_rng_fwd_%s.c" r isa
             , [ string_of_int r; "--hc2hc"; "--ranged"; "--t1s"; "--isa"; isa; "--su" ] )
           ; ( Printf.sprintf "radix%d_hc2c_nat_rng_fwd_%s.c" r isa
             , [ string_of_int r
               ; "--hc2c-nat"
               ; "--ranged"
               ; "--t1s"
               ; "--isa"
               ; isa
               ; "--su"
               ] )
           ])
        radices
  | "trig-avx2" | "trig-avx512" ->
    (* Real-to-real trig family (notebook section 51): lean 3-arg ABI
     * (in, out, K), constant hoisting on, consumed by core/dct.h-style
     * plan shells. rdft deferred: complex output, generic ABI. *)
    let isa = if quadrant = "trig-avx2" then "avx2" else "avx512" in
    let kind_sizes =
      [ "dct2", [ 8; 16; 32; 64 ]
      ; "dct3", [ 8; 16; 32; 64 ]
      ; "dct4", [ 8; 16; 32; 64 ]
      ; "dst2", [ 8; 16; 32; 64 ]
      ; "dst3", [ 8; 16; 32; 64 ]
      ; "dst4", [ 8; 16; 32; 64 ]
      ; "dht", [ 8; 16; 32; 64 ]
      ; (* Boundary kinds run at their logical-extension sizes:
         dct1 at N needs M = 2(N-1) radix-coverable -> N = 2^k+1
         (Chebyshev grids); dst1 needs M = 2(N+1) -> N = 2^k-1. *)
        "dct1", [ 5; 9; 17; 33 ]
      ; "dst1", [ 3; 7; 15; 31 ]
      ]
    in
    List.concat_map
      (fun (kind, sizes) ->
         List.map
           (fun n ->
              ( Printf.sprintf "radix%d_%s_%s.c" n kind isa
              , [ string_of_int n; "--" ^ kind; "--isa"; isa; "--su" ] ))
           sizes)
      kind_sizes
  | "c2r-avx2" | "c2r-avx512" ->
    (* Native real-cascade BACKWARD family (section 62, the inverse of the
     * rfft forward quadrant). FFTW runs hc2r as apply_DIF + sign-flipped
     * twiddles: r2cb leaf (halfcomplex -> real) + hc2hc DIF backward stages.
     * The c2r executor (core/c2r.h) calls exactly these; the matrix gate
     * (benchmarks/gate_c2r_matrix.c) proved them across nf=1..4 incl. (8,32).
     * --t1s is REQUIRED (scalar-broadcast twiddles, doc 60 gotcha) or the
     * codelet reads the wrong twiddle memory. radices mirror the forward
     * quadrant; radix-32 is the big leaf for the (8,32) MKL-beating plan. *)
    let isa = if quadrant = "c2r-avx2" then "avx2" else "avx512" in
    let radices = [ 2; 3; 4; 5; 7; 8; 16 ] in
    let leaf_only = [ 32 ] in
    List.map
      (fun r ->
         ( Printf.sprintf "radix%d_r2cb_%s.c" r isa
         , [ string_of_int r; "--r2cb"; "--isa"; isa; "--su" ] ))
      leaf_only
    @ List.concat_map
        (fun r ->
           [ ( Printf.sprintf "radix%d_r2cb_%s.c" r isa
             , [ string_of_int r; "--r2cb"; "--isa"; isa; "--su" ] )
           ; ( Printf.sprintf "radix%d_hc2hc_dif_bwd_%s.c" r isa
             , [ string_of_int r
               ; "--hc2hc"
               ; "--dif"
               ; "--bwd"
               ; "--t1s"
               ; "--isa"
               ; isa
               ; "--su"
               ] )
           ; ( Printf.sprintf "radix%d_hc2hc_dif_log3_bwd_%s.c" r isa
             , [ string_of_int r
               ; "--hc2hc"
               ; "--dif"
               ; "--bwd"
               ; "--log3"
               ; "--t1s"
               ; "--isa"
               ; isa
               ; "--su"
               ] )
           ; ( Printf.sprintf "radix%d_hc2hc_dif_rng_bwd_%s.c" r isa
             , [ string_of_int r
               ; "--hc2hc"
               ; "--dif"
               ; "--bwd"
               ; "--ranged"
               ; "--t1s"
               ; "--isa"
               ; isa
               ; "--su"
               ] )
           ; (* c2r NATURAL INITIATOR (stage-0 split-input, inverse of the rfft
              * forward hc2c_nat terminator): reads the SPLIT half-spectrum and
              * feeds the packed c2r cascade with no repack. --bwd --dif matches
              * the cascade orientation; verified to invert the forward at 7e-15. *)
             ( Printf.sprintf "radix%d_hc2c_nat_log3_bwd_%s.c" r isa
             , [ string_of_int r
               ; "--hc2c-nat"
               ; "--bwd"
               ; "--dif"
               ; "--log3"
               ; "--t1s"
               ; "--isa"
               ; isa
               ; "--su"
               ] )
           ; ( Printf.sprintf "radix%d_hc2c_nat_bwd_%s.c" r isa
             , [ string_of_int r
               ; "--hc2c-nat"
               ; "--bwd"
               ; "--dif"
               ; "--t1s"
               ; "--isa"
               ; isa
               ; "--su"
               ] )
           ; ( Printf.sprintf "radix%d_hc2c_nat_rng_bwd_%s.c" r isa
             , [ string_of_int r
               ; "--hc2c-nat"
               ; "--bwd"
               ; "--dif"
               ; "--ranged"
               ; "--t1s"
               ; "--isa"
               ; isa
               ; "--su"
               ] )
           ])
        radices
  | q -> failwith ("Corpus.files: unknown quadrant " ^ q)
;;

let quadrants =
  [ "inplace-avx2"
  ; "inplace-avx512"
  ; "oop-avx2"
  ; "oop-avx512"
  ; "strided-avx2"
  ; "strided-avx512"
  ; "trig-avx2"
  ; "trig-avx512"
  ; "rfft-avx2"
  ; "rfft-avx512"
  ; "c2r-avx2"
  ; "c2r-avx512"
  ; "oop-edges-avx2"
  ; "oop-edges-avx512"
  ; "strided-r2c-avx2"
  ; "strided-r2c-avx512"
  ; "zil-boundary"
  ; "zil-pure"
  ]
;;

(* directory under codelets/ for each quadrant *)
let dir_of_quadrant (q : string) : string =
  match q with
  | "oop-edges-avx2" -> "oop/avx2"
  | "oop-edges-avx512" -> "oop/avx512"
  | "strided-r2c-avx2" -> "strided/avx2"
  | "strided-r2c-avx512" -> "strided/avx512"
  | "zil-boundary" -> "zil/avx2/boundary_split"
  | "zil-pure" -> "zil/avx2/pure_il"
  | _ ->
    (match String.split_on_char '-' q with
     | [ fam; isa ] -> fam ^ "/" ^ isa
     | _ -> failwith ("Corpus.dir_of_quadrant: " ^ q))
;;

(* expected_counts: DELETED at M10 — zero consumers repo-wide. *)

(* ── M10: the typed layer — the corpus AS Codelet.t values ── *)

type cell =
  { file : string
  ; c : Codelet.t
  }

let strip_emit_c (argv : string list) : string list =
  match List.rev argv with
  | "--emit-c" :: rest -> List.rev rest
  | _ -> failwith "Corpus: canonical argv does not end with --emit-c"
;;

let cells_of_quadrant (q : string) : cell list =
  List.map
    (fun (file, tail) ->
       let full = tail @ [ "--emit-c" ] in
       let c =
         try Codelet.of_argv full with
         | Codelet.Parse_error m ->
           failwith (Printf.sprintf "Corpus law (%s/%s): does not parse: %s" q file m)
       in
       let rt = Codelet.to_argv c in
       if rt <> full
       then
         failwith
           (Printf.sprintf
              "Corpus law (%s/%s): round-trip drift.\n  matrix: %s\n  to_argv: %s"
              q
              file
              (String.concat " " full)
              (String.concat " " rt));
       { file; c })
    (matrix_files q)
;;

(* One lazy block: construct every quadrant, then the uniqueness law.
 * Forced by cells/files — i.e. by gen_set and the registry emitters,
 * never by single-codelet gen_radix runs. *)
let corpus : (string * cell list) list Lazy.t =
  lazy
    (let per_q = List.map (fun q -> q, cells_of_quadrant q) quadrants in
     let seen_file = Hashtbl.create 1201 in
     let seen_argv = Hashtbl.create 1201 in
     List.iter
       (fun (q, cs) ->
          let dir = dir_of_quadrant q in
          List.iter
            (fun { file; c } ->
               let path = dir ^ "/" ^ file in
               (match Hashtbl.find_opt seen_file path with
                | Some q0 ->
                  failwith
                    (Printf.sprintf
                       "Corpus law: duplicate file %s (quadrants %s and %s)"
                       path
                       q0
                       q)
                | None -> Hashtbl.add seen_file path q);
               let key = String.concat " " (Codelet.to_argv c) in
               match Hashtbl.find_opt seen_argv key with
               | Some f0 ->
                 failwith
                   (Printf.sprintf
                      "Corpus law: %s and %s share one canonical argv: %s"
                      f0
                      path
                      key)
               | None -> Hashtbl.add seen_argv key path)
            cs)
       per_q;
     per_q)
;;

let cells (q : string) : cell list =
  match List.assoc_opt q (Lazy.force corpus) with
  | Some cs -> cs
  | None -> failwith ("Corpus.cells: unknown quadrant " ^ q)
;;

(* The public files: DERIVED from the descriptor (to_argv), not from the
 * matrix — the descriptor is the source of truth; the round-trip law
 * makes this byte-equal to the matrix output. *)
let files (q : string) : (string * string list) list =
  List.map (fun { file; c } -> file, strip_emit_c (Codelet.to_argv c)) (cells q)
;;
