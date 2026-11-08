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
 * Consumers: gen_set (tree regen), the registry emitters (files).
 * ⚠ The emitted registry headers still
 * SAY "Coverage.files" in their comment banner — kept byte-identical
 * deliberately at M10; the wording updates with the first REAL registry
 * change (a coverage raise), never as a rename side effect.
 * cil/zil enter the corpus at M12a (state fix first), never before.
 *)

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
(* zturn-ts (2026-11-04): the SPLIT library's ZTURN-T stage kernels, quadrant
   ztt-ISA -> codelets/split/ztt/ISA. Two families x four kinds x radix 4/8 x
   two directions = 32 cells, plus the odd radices 3/5/7/9/15 at the mids of
   both families and at the lane-major family's ends (tv0tp, tvlfi: no lane
   lattice there), both directions: the staged forms of the 2^a*odd band and
   of the lane-major flat DIT (2026-11-06): zts = K=1 (split planes at the edges, the block
   plane inside), ztv = lane-major K=VW (lane = transform). The fused drivers
   in generated/fused_codelets (zts_drivers_ and ztv_drivers_ files) inline
   these bodies; cascade_z.ml emits both. File radixR_FAM_KIND[_bwd]_ISA.c,
   symbol radixR_FAM_KIND_fwd|bwd_ISA. *)
let zts_cells (isa : string) : (string * string list) list =
  List.concat_map
    (fun (fam, pfx) ->
       List.concat_map
         (fun base ->
            List.concat_map
              (fun r ->
                 List.map
                   (fun bwd ->
                      let kind = pfx ^ String.sub base 1 (String.length base - 1) ^ if bwd then "b" else "" in
                      ( Printf.sprintf "radix%d_%s_%s%s_%s.c" r fam base (if bwd then "_bwd" else "") isa
                      , [ string_of_int r; "--zp-" ^ kind; "--isa"; isa; "--uarch"; "raptor_lake_" ^ isa ] ))
                   [ false; true ])
              (* the mids carry no lane lattice: the odd radices of the 2^a*odd band
                 (ztt_odd_design.md) join them, 2026-11-06; the ends stay 4/8 *)
              (if base = "tmg" || (fam = "ztv" && base <> "tlf") then [ 4; 8; 3; 5; 7; 9; 15 ] else [ 4; 8 ]))
         [ "t0tp"; "tmg"; "tlf"; "tlfi" ])
    [ "zts", "ts"; "ztv", "tv" ]
;;

let zil_boundary_cells : (string * string list) list =
  (* the ZTURN-T stage kernels (natural: t0tp/tmg/tlf/tlfi; plain: t0d/tmgd/tld/tldb;
     the odd mids tmg/tmgb/tmgd at 3/5/7/9/15) and the msz/mszt odd mids of il2p —
     every kind cascade_z.ml emits. *)
  [ ( "radix4_z_t0tp_avx2.c"
    , [ "4"; "--zp-t0tp"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix4_z_t0tp_bwd_avx2.c"
    , [ "4"; "--zp-t0tpb"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix8_z_t0tp_avx2.c"
    , [ "8"; "--zp-t0tp"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix8_z_t0tp_bwd_avx2.c"
    , [ "8"; "--zp-t0tpb"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix4_z_tmg_avx2.c"
    , [ "4"; "--zp-tmg"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix4_z_tmg_bwd_avx2.c"
    , [ "4"; "--zp-tmgb"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix8_z_tmg_avx2.c"
    , [ "8"; "--zp-tmg"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix8_z_tmg_bwd_avx2.c"
    , [ "8"; "--zp-tmgb"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix4_z_tlf_avx2.c"
    , [ "4"; "--zp-tlf"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix4_z_tlf_bwd_avx2.c"
    , [ "4"; "--zp-tlfb"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix8_z_tlf_avx2.c"
    , [ "8"; "--zp-tlf"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix8_z_tlf_bwd_avx2.c"
    , [ "8"; "--zp-tlfb"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix4_z_tlfi_avx2.c"
    , [ "4"; "--zp-tlfi"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix4_z_tlfi_bwd_avx2.c"
    , [ "4"; "--zp-tlfib"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix8_z_tlfi_avx2.c"
    , [ "8"; "--zp-tlfi"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix8_z_tlfi_bwd_avx2.c"
    , [ "8"; "--zp-tlfib"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix4_z_t0d_avx2.c"
    , [ "4"; "--zp-t0d"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix8_z_t0d_avx2.c"
    , [ "8"; "--zp-t0d"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix4_z_tmgd_avx2.c"
    , [ "4"; "--zp-tmgd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix8_z_tmgd_avx2.c"
    , [ "8"; "--zp-tmgd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix4_z_tld_avx2.c"
    , [ "4"; "--zp-tld"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix4_z_tld_bwd_avx2.c"
    , [ "4"; "--zp-tldb"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix8_z_tld_avx2.c"
    , [ "8"; "--zp-tld"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix8_z_tld_bwd_avx2.c"
    , [ "8"; "--zp-tldb"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix3_z_tmg_avx2.c"
    , [ "3"; "--zp-tmg"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix3_z_tmg_bwd_avx2.c"
    , [ "3"; "--zp-tmgb"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix3_z_tmgd_avx2.c"
    , [ "3"; "--zp-tmgd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix5_z_tmg_avx2.c"
    , [ "5"; "--zp-tmg"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix5_z_tmg_bwd_avx2.c"
    , [ "5"; "--zp-tmgb"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix5_z_tmgd_avx2.c"
    , [ "5"; "--zp-tmgd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix7_z_tmg_avx2.c"
    , [ "7"; "--zp-tmg"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix7_z_tmg_bwd_avx2.c"
    , [ "7"; "--zp-tmgb"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix7_z_tmgd_avx2.c"
    , [ "7"; "--zp-tmgd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix9_z_tmg_avx2.c"
    , [ "9"; "--zp-tmg"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix9_z_tmg_bwd_avx2.c"
    , [ "9"; "--zp-tmgb"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix9_z_tmgd_avx2.c"
    , [ "9"; "--zp-tmgd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix15_z_tmg_avx2.c"
    , [ "15"; "--zp-tmg"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix15_z_tmg_bwd_avx2.c"
    , [ "15"; "--zp-tmgb"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix15_z_tmgd_avx2.c"
    , [ "15"; "--zp-tmgd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix3_z_msz_avx2.c"
    , [ "3"; "--zp-msz"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix5_z_msz_avx2.c"
    , [ "5"; "--zp-msz"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix7_z_msz_avx2.c"
    , [ "7"; "--zp-msz"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix9_z_msz_avx2.c"
    , [ "9"; "--zp-msz"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix15_z_msz_avx2.c"
    , [ "15"; "--zp-msz"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix3_z_msz_bwd_avx2.c"
    , [ "3"; "--zp-mszb"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix5_z_msz_bwd_avx2.c"
    , [ "5"; "--zp-mszb"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix7_z_msz_bwd_avx2.c"
    , [ "7"; "--zp-mszb"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix9_z_msz_bwd_avx2.c"
    , [ "9"; "--zp-mszb"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix15_z_msz_bwd_avx2.c"
    , [ "15"; "--zp-mszb"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix3_z_mszt_bwd_avx2.c"
    , [ "3"; "--zp-mszt"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix5_z_mszt_bwd_avx2.c"
    , [ "5"; "--zp-mszt"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix7_z_mszt_bwd_avx2.c"
    , [ "7"; "--zp-mszt"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix9_z_mszt_bwd_avx2.c"
    , [ "9"; "--zp-mszt"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix15_z_mszt_bwd_avx2.c"
    , [ "15"; "--zp-mszt"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ]
;;

let zil_pure_cells : (string * string list) list =
  [ ( "radix9_z_n1t_ct_bwd_avx2.c"
    , [ "9"; "--cil-n1t"; "--cil-oddct"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix9_z_t2_ct_bwd_avx2.c"
    , [ "9"; "--cil-t2"; "--cil-oddct"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix15_z_n1t_ct_avx2.c"
    , [ "15"; "--cil-n1t"; "--cil-oddct"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix15_z_n1t_ct_bwd_avx2.c"
    , [ "15"; "--cil-n1t"; "--cil-oddct"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix15_z_t2_ct_avx2.c"
    , [ "15"; "--cil-t2"; "--cil-oddct"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix15_z_t2_ct_bwd_avx2.c"
    , [ "15"; "--cil-t2"; "--cil-oddct"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix21_z_n1t_ct_avx2.c"
    , [ "21"; "--cil-n1t"; "--cil-oddct"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix21_z_n1t_ct_bwd_avx2.c"
    , [ "21"; "--cil-n1t"; "--cil-oddct"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix21_z_t2_ct_avx2.c"
    , [ "21"; "--cil-t2"; "--cil-oddct"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix21_z_t2_ct_bwd_avx2.c"
    , [ "21"; "--cil-t2"; "--cil-oddct"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix25_z_n1t_ct_avx2.c"
    , [ "25"; "--cil-n1t"; "--cil-oddct"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix25_z_n1t_ct_bwd_avx2.c"
    , [ "25"; "--cil-n1t"; "--cil-oddct"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix25_z_t2_ct_avx2.c"
    , [ "25"; "--cil-t2"; "--cil-oddct"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix25_z_t2_ct_bwd_avx2.c"
    , [ "25"; "--cil-t2"; "--cil-oddct"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix27_z_n1t_ct_avx2.c"
    , [ "27"; "--cil-n1t"; "--cil-oddct"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix27_z_n1t_ct_bwd_avx2.c"
    , [ "27"; "--cil-n1t"; "--cil-oddct"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix27_z_t2_ct_avx2.c"
    , [ "27"; "--cil-t2"; "--cil-oddct"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix27_z_t2_ct_bwd_avx2.c"
    , [ "27"; "--cil-t2"; "--cil-oddct"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix10_z_n1c_avx2.c"
    , [ "10"; "--cil-n1c"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix10_z_n1c_bwd_avx2.c"
    , [ "10"; "--cil-n1c"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix10_z_n1_avx2.c"
    , [ "10"; "--cil-n1"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix10_z_n1_bwd_avx2.c"
    , [ "10"; "--cil-n1"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix14_z_n1c_avx2.c"
    , [ "14"; "--cil-n1c"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix14_z_n1c_bwd_avx2.c"
    , [ "14"; "--cil-n1c"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix14_z_n1_avx2.c"
    , [ "14"; "--cil-n1"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix14_z_n1_bwd_avx2.c"
    , [ "14"; "--cil-n1"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix22_z_n1c_avx2.c"
    , [ "22"; "--cil-n1c"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix22_z_n1c_bwd_avx2.c"
    , [ "22"; "--cil-n1c"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix22_z_n1_avx2.c"
    , [ "22"; "--cil-n1"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix22_z_n1_bwd_avx2.c"
    , [ "22"; "--cil-n1"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix26_z_n1c_avx2.c"
    , [ "26"; "--cil-n1c"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix26_z_n1c_bwd_avx2.c"
    , [ "26"; "--cil-n1c"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix26_z_n1_avx2.c"
    , [ "26"; "--cil-n1"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix26_z_n1_bwd_avx2.c"
    , [ "26"; "--cil-n1"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
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
  ; ( "radix12_z_n1c_avx2.c"
    , [ "12"; "--cil-n1c"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix12_z_n1c_bwd_avx2.c"
    , [ "12"; "--cil-n1c"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
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
  ; ( "radix14_z_n1t_avx2.c"
    , [ "14"; "--cil-n1t"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix14_z_n1t_bwd_avx2.c"
    , [ "14"; "--cil-n1t"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix14_z_t2_avx2.c"
    , [ "14"; "--cil-t2"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix14_z_t2_bwd_avx2.c"
    , [ "14"; "--cil-t2"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix14_z_t2t_bwd_avx2.c"
    , [ "14"
      ; "--cil-t2"
      ; "--cil-turnst"
      ; "--cil-bwd"
      ; "--isa"
      ; "avx2"
      ; "--uarch"
      ; "raptor_lake_avx2"
      ] )
  ; ( "radix14_z_t2tg_bwd_avx2.c"
    , [ "14"
      ; "--cil-t2"
      ; "--cil-turnst-gs"
      ; "--cil-bwd"
      ; "--isa"
      ; "avx2"
      ; "--uarch"
      ; "raptor_lake_avx2"
      ] )
  ; ( "radix20_z_n1_bwd_avx2.c"
    , [ "20"; "--cil-n1"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix20_z_n1t_avx2.c"
    , [ "20"; "--cil-n1t"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix20_z_n1t_bwd_avx2.c"
    , [ "20"; "--cil-n1t"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix20_z_t2_avx2.c"
    , [ "20"; "--cil-t2"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix20_z_t2_bwd_avx2.c"
    , [ "20"; "--cil-t2"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix20_z_t2t_bwd_avx2.c"
    , [ "20"
      ; "--cil-t2"
      ; "--cil-turnst"
      ; "--cil-bwd"
      ; "--isa"
      ; "avx2"
      ; "--uarch"
      ; "raptor_lake_avx2"
      ] )
  ; ( "radix20_z_t2tg_bwd_avx2.c"
    , [ "20"
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
  ; ( "radix15_z_n1t_avx2.c"
    , [ "15"; "--cil-n1t"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix15_z_n1t_bwd_avx2.c"
    , [ "15"; "--cil-n1t"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix15_z_t2_avx2.c"
    , [ "15"; "--cil-t2"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix15_z_t2_bwd_avx2.c"
    , [ "15"; "--cil-t2"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
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
  ; ( "radix16_z_n1_avx2.c"
    , [ "16"; "--cil-n1"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
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
  ; ( "radix23_z_n1_avx2.c"
    , [ "23"; "--cil-n1"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix43_z_n1_avx2.c"
    , [ "43"; "--cil-n1"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix47_z_n1_avx2.c"
    , [ "47"; "--cil-n1"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix37_z_n1_avx2.c"
    , [ "37"; "--cil-n1"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix41_z_n1_avx2.c"
    , [ "41"; "--cil-n1"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix29_z_n1_avx2.c"
    , [ "29"; "--cil-n1"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix31_z_n1_avx2.c"
    , [ "31"; "--cil-n1"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix19_z_n1_bwd_avx2.c"
    , [ "19"; "--cil-n1"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix23_z_n1_bwd_avx2.c"
    , [ "23"; "--cil-n1"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix43_z_n1_bwd_avx2.c"
    , [ "43"; "--cil-n1"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix47_z_n1_bwd_avx2.c"
    , [ "47"; "--cil-n1"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix37_z_n1_bwd_avx2.c"
    , [ "37"; "--cil-n1"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix41_z_n1_bwd_avx2.c"
    , [ "41"; "--cil-n1"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix29_z_n1_bwd_avx2.c"
    , [ "29"; "--cil-n1"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix31_z_n1_bwd_avx2.c"
    , [ "31"; "--cil-n1"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix19_z_n1t_avx2.c"
    , [ "19"; "--cil-n1t"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix23_z_n1t_avx2.c"
    , [ "23"; "--cil-n1t"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix43_z_n1t_avx2.c"
    , [ "43"; "--cil-n1t"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix47_z_n1t_avx2.c"
    , [ "47"; "--cil-n1t"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix37_z_n1t_avx2.c"
    , [ "37"; "--cil-n1t"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix41_z_n1t_avx2.c"
    , [ "41"; "--cil-n1t"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix29_z_n1t_avx2.c"
    , [ "29"; "--cil-n1t"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix31_z_n1t_avx2.c"
    , [ "31"; "--cil-n1t"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix19_z_n1t_bwd_avx2.c"
    , [ "19"; "--cil-n1t"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix23_z_n1t_bwd_avx2.c"
    , [ "23"; "--cil-n1t"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix43_z_n1t_bwd_avx2.c"
    , [ "43"; "--cil-n1t"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix47_z_n1t_bwd_avx2.c"
    , [ "47"; "--cil-n1t"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix37_z_n1t_bwd_avx2.c"
    , [ "37"; "--cil-n1t"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix41_z_n1t_bwd_avx2.c"
    , [ "41"; "--cil-n1t"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix29_z_n1t_bwd_avx2.c"
    , [ "29"; "--cil-n1t"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix31_z_n1t_bwd_avx2.c"
    , [ "31"; "--cil-n1t"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix19_z_t2_avx2.c"
    , [ "19"; "--cil-t2"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix23_z_t2_avx2.c"
    , [ "23"; "--cil-t2"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix43_z_t2_avx2.c"
    , [ "43"; "--cil-t2"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix47_z_t2_avx2.c"
    , [ "47"; "--cil-t2"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix37_z_t2_avx2.c"
    , [ "37"; "--cil-t2"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix41_z_t2_avx2.c"
    , [ "41"; "--cil-t2"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix29_z_t2_avx2.c"
    , [ "29"; "--cil-t2"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix31_z_t2_avx2.c"
    , [ "31"; "--cil-t2"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix19_z_t2_bwd_avx2.c"
    , [ "19"; "--cil-t2"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix23_z_t2_bwd_avx2.c"
    , [ "23"; "--cil-t2"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix43_z_t2_bwd_avx2.c"
    , [ "43"; "--cil-t2"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix47_z_t2_bwd_avx2.c"
    , [ "47"; "--cil-t2"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix37_z_t2_bwd_avx2.c"
    , [ "37"; "--cil-t2"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix41_z_t2_bwd_avx2.c"
    , [ "41"; "--cil-t2"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix29_z_t2_bwd_avx2.c"
    , [ "29"; "--cil-t2"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix31_z_t2_bwd_avx2.c"
    , [ "31"; "--cil-t2"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
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
  ; ( "radix23_z_t2t_bwd_avx2.c"
    , [ "23"
      ; "--cil-t2"
      ; "--cil-turnst"
      ; "--cil-bwd"
      ; "--isa"
      ; "avx2"
      ; "--uarch"
      ; "raptor_lake_avx2"
      ] )
  ; ( "radix43_z_t2t_bwd_avx2.c"
    , [ "43"
      ; "--cil-t2"
      ; "--cil-turnst"
      ; "--cil-bwd"
      ; "--isa"
      ; "avx2"
      ; "--uarch"
      ; "raptor_lake_avx2"
      ] )
  ; ( "radix47_z_t2t_bwd_avx2.c"
    , [ "47"
      ; "--cil-t2"
      ; "--cil-turnst"
      ; "--cil-bwd"
      ; "--isa"
      ; "avx2"
      ; "--uarch"
      ; "raptor_lake_avx2"
      ] )
  ; ( "radix37_z_t2t_bwd_avx2.c"
    , [ "37"
      ; "--cil-t2"
      ; "--cil-turnst"
      ; "--cil-bwd"
      ; "--isa"
      ; "avx2"
      ; "--uarch"
      ; "raptor_lake_avx2"
      ] )
  ; ( "radix41_z_t2t_bwd_avx2.c"
    , [ "41"
      ; "--cil-t2"
      ; "--cil-turnst"
      ; "--cil-bwd"
      ; "--isa"
      ; "avx2"
      ; "--uarch"
      ; "raptor_lake_avx2"
      ] )
  ; ( "radix29_z_t2t_bwd_avx2.c"
    , [ "29"
      ; "--cil-t2"
      ; "--cil-turnst"
      ; "--cil-bwd"
      ; "--isa"
      ; "avx2"
      ; "--uarch"
      ; "raptor_lake_avx2"
      ] )
  ; ( "radix31_z_t2t_bwd_avx2.c"
    , [ "31"
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
  ; ( "radix23_z_t2tg_bwd_avx2.c"
    , [ "23"
      ; "--cil-t2"
      ; "--cil-turnst-gs"
      ; "--cil-bwd"
      ; "--isa"
      ; "avx2"
      ; "--uarch"
      ; "raptor_lake_avx2"
      ] )
  ; ( "radix43_z_t2tg_bwd_avx2.c"
    , [ "43"
      ; "--cil-t2"
      ; "--cil-turnst-gs"
      ; "--cil-bwd"
      ; "--isa"
      ; "avx2"
      ; "--uarch"
      ; "raptor_lake_avx2"
      ] )
  ; ( "radix47_z_t2tg_bwd_avx2.c"
    , [ "47"
      ; "--cil-t2"
      ; "--cil-turnst-gs"
      ; "--cil-bwd"
      ; "--isa"
      ; "avx2"
      ; "--uarch"
      ; "raptor_lake_avx2"
      ] )
  ; ( "radix37_z_t2tg_bwd_avx2.c"
    , [ "37"
      ; "--cil-t2"
      ; "--cil-turnst-gs"
      ; "--cil-bwd"
      ; "--isa"
      ; "avx2"
      ; "--uarch"
      ; "raptor_lake_avx2"
      ] )
  ; ( "radix41_z_t2tg_bwd_avx2.c"
    , [ "41"
      ; "--cil-t2"
      ; "--cil-turnst-gs"
      ; "--cil-bwd"
      ; "--isa"
      ; "avx2"
      ; "--uarch"
      ; "raptor_lake_avx2"
      ] )
  ; ( "radix29_z_t2tg_bwd_avx2.c"
    , [ "29"
      ; "--cil-t2"
      ; "--cil-turnst-gs"
      ; "--cil-bwd"
      ; "--isa"
      ; "avx2"
      ; "--uarch"
      ; "raptor_lake_avx2"
      ] )
  ; ( "radix31_z_t2tg_bwd_avx2.c"
    , [ "31"
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
  ; ( "radix21_z_n1t_avx2.c"
    , [ "21"; "--cil-n1t"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix21_z_n1t_bwd_avx2.c"
    , [ "21"; "--cil-n1t"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix21_z_t2_avx2.c"
    , [ "21"; "--cil-t2"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix21_z_t2_bwd_avx2.c"
    , [ "21"; "--cil-t2"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
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
  ; ( "radix25_z_n1t_avx2.c"
    , [ "25"; "--cil-n1t"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix25_z_n1t_bwd_avx2.c"
    , [ "25"; "--cil-n1t"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix25_z_t2_avx2.c"
    , [ "25"; "--cil-t2"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix25_z_t2_bwd_avx2.c"
    , [ "25"; "--cil-t2"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
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
  ; ( "radix27_z_n1t_avx2.c"
    , [ "27"; "--cil-n1t"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix27_z_n1t_bwd_avx2.c"
    , [ "27"; "--cil-n1t"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix27_z_t2_avx2.c"
    , [ "27"; "--cil-t2"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix27_z_t2_bwd_avx2.c"
    , [ "27"; "--cil-t2"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
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
  ; ( "radix32_z_n1_avx2.c"
    , [ "32"; "--cil-n1"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
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
  ; ( "radix2_z_n1c_avx2.c"
    , [ "2"; "--cil-n1c"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix2_z_n1c_bwd_avx2.c"
    , [ "2"; "--cil-n1c"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix2_z_n1_avx2.c"
    , [ "2"; "--cil-n1"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix2_z_n1_bwd_avx2.c"
    , [ "2"; "--cil-n1"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
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
  ; ( "radix4_z_n1_avx2.c"
    , [ "4"; "--cil-n1"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
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
  ; ( "radix64_z_n1_avx2.c"
    , [ "64"; "--cil-n1"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
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
  ; ( "radix6_z_n1c_avx2.c"
    , [ "6"; "--cil-n1c"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix6_z_n1c_bwd_avx2.c"
    , [ "6"; "--cil-n1c"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
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
  ; ( "radix8_z_n1_avx2.c"
    , [ "8"; "--cil-n1"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
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
  ; ( "radix9_z_n1t_avx2.c"
    , [ "9"; "--cil-n1t"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix9_z_n1t_bwd_avx2.c"
    , [ "9"; "--cil-n1t"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix9_z_t2_avx2.c"
    , [ "9"; "--cil-t2"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix9_z_t2_bwd_avx2.c"
    , [ "9"; "--cil-t2"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
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
  ; ( "radix32_z_t2bt48_bwd_avx2.c"
    , [ "32"; "--cil-t2"; "--cil-blocked"; "--cil-split"; "4.8"; "--cil-turnst"; "--cil-form-tag"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix32_z_n1b48_bwd_avx2.c"
    , [ "32"; "--cil-n1"; "--cil-blocked"; "--cil-split"; "4.8"; "--cil-form-tag"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix32_z_t2bt216_bwd_avx2.c"
    , [ "32"; "--cil-t2"; "--cil-blocked"; "--cil-split"; "2.16"; "--cil-turnst"; "--cil-form-tag"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix32_z_n1b216_bwd_avx2.c"
    , [ "32"; "--cil-n1"; "--cil-blocked"; "--cil-split"; "2.16"; "--cil-form-tag"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix64_z_t2bt88_bwd_avx2.c"
    , [ "64"; "--cil-t2"; "--cil-blocked"; "--cil-split"; "8.8"; "--cil-turnst"; "--cil-form-tag"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix64_z_n1b88_bwd_avx2.c"
    , [ "64"; "--cil-n1"; "--cil-blocked"; "--cil-split"; "8.8"; "--cil-form-tag"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix64_z_t2bt416_bwd_avx2.c"
    , [ "64"; "--cil-t2"; "--cil-blocked"; "--cil-split"; "4.16"; "--cil-turnst"; "--cil-form-tag"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix64_z_n1b416_bwd_avx2.c"
    , [ "64"; "--cil-n1"; "--cil-blocked"; "--cil-split"; "4.16"; "--cil-form-tag"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix64_z_n1tb88_avx2.c"
    , [ "64"; "--cil-n1t"; "--cil-blocked"; "--cil-split"; "8.8"; "--cil-form-tag"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix64_z_n1tb416_avx2.c"
    , [ "64"; "--cil-n1t"; "--cil-blocked"; "--cil-split"; "4.16"; "--cil-form-tag"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix64_z_t2b88_avx2.c"
    , [ "64"; "--cil-t2"; "--cil-blocked"; "--cil-split"; "8.8"; "--cil-form-tag"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix64_z_t2b416_avx2.c"
    , [ "64"; "--cil-t2"; "--cil-blocked"; "--cil-split"; "4.16"; "--cil-form-tag"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix15_z_t2t_ct_bwd_avx2.c"
    , [ "15"; "--cil-t2"; "--cil-oddct"; "--cil-turnst"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix15_z_n1_ct_bwd_avx2.c"
    , [ "15"; "--cil-n1"; "--cil-oddct"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix21_z_t2t_ct_bwd_avx2.c"
    , [ "21"; "--cil-t2"; "--cil-oddct"; "--cil-turnst"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix21_z_n1_ct_bwd_avx2.c"
    , [ "21"; "--cil-n1"; "--cil-oddct"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix25_z_t2t_ct_bwd_avx2.c"
    , [ "25"; "--cil-t2"; "--cil-oddct"; "--cil-turnst"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix25_z_n1_ct_bwd_avx2.c"
    , [ "25"; "--cil-n1"; "--cil-oddct"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix27_z_t2t_ct_bwd_avx2.c"
    , [ "27"; "--cil-t2"; "--cil-oddct"; "--cil-turnst"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix27_z_n1_ct_bwd_avx2.c"
    , [ "27"; "--cil-n1"; "--cil-oddct"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix4_z_n1c_avx2.c"
    , [ "4"; "--cil-n1c"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix4_z_n1c_bwd_avx2.c"
    , [ "4"; "--cil-n1c"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix8_z_n1c_avx2.c"
    , [ "8"; "--cil-n1c"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix8_z_n1c_bwd_avx2.c"
    , [ "8"; "--cil-n1c"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix16_z_n1c_avx2.c"
    , [ "16"; "--cil-n1c"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix16_z_n1c_bwd_avx2.c"
    , [ "16"; "--cil-n1c"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix32_z_n1c_avx2.c"
    , [ "32"; "--cil-n1c"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix32_z_n1c_bwd_avx2.c"
    , [ "32"; "--cil-n1c"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix64_z_n1c_avx2.c"
    , [ "64"; "--cil-n1c"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix64_z_n1c_bwd_avx2.c"
    , [ "64"; "--cil-n1c"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix4_z_t2c_avx2.c"
    , [ "4"; "--cil-t2c"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix4_z_t2c_bwd_avx2.c"
    , [ "4"; "--cil-t2c"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix8_z_t2c_avx2.c"
    , [ "8"; "--cil-t2c"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix8_z_t2c_bwd_avx2.c"
    , [ "8"; "--cil-t2c"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix3_z_t2c_avx2.c"
    , [ "3"; "--cil-t2c"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix3_z_t2c_bwd_avx2.c"
    , [ "3"; "--cil-t2c"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix3_z_n1c_avx2.c"
    , [ "3"; "--cil-n1c"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix3_z_n1c_bwd_avx2.c"
    , [ "3"; "--cil-n1c"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix5_z_t2c_avx2.c"
    , [ "5"; "--cil-t2c"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix5_z_t2c_bwd_avx2.c"
    , [ "5"; "--cil-t2c"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix5_z_n1c_avx2.c"
    , [ "5"; "--cil-n1c"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix5_z_n1c_bwd_avx2.c"
    , [ "5"; "--cil-n1c"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix7_z_t2c_avx2.c"
    , [ "7"; "--cil-t2c"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix7_z_t2c_bwd_avx2.c"
    , [ "7"; "--cil-t2c"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix7_z_n1c_avx2.c"
    , [ "7"; "--cil-n1c"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix7_z_n1c_bwd_avx2.c"
    , [ "7"; "--cil-n1c"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix9_z_t2c_avx2.c"
    , [ "9"; "--cil-t2c"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix9_z_t2c_bwd_avx2.c"
    , [ "9"; "--cil-t2c"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix9_z_n1c_avx2.c"
    , [ "9"; "--cil-n1c"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix9_z_n1c_bwd_avx2.c"
    , [ "9"; "--cil-n1c"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix11_z_t2c_avx2.c"
    , [ "11"; "--cil-t2c"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix11_z_t2c_bwd_avx2.c"
    , [ "11"; "--cil-t2c"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix11_z_n1c_avx2.c"
    , [ "11"; "--cil-n1c"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix11_z_n1c_bwd_avx2.c"
    , [ "11"; "--cil-n1c"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix13_z_t2c_avx2.c"
    , [ "13"; "--cil-t2c"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix13_z_t2c_bwd_avx2.c"
    , [ "13"; "--cil-t2c"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix13_z_n1c_avx2.c"
    , [ "13"; "--cil-n1c"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix13_z_n1c_bwd_avx2.c"
    , [ "13"; "--cil-n1c"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix15_z_t2c_avx2.c"
    , [ "15"; "--cil-t2c"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix15_z_t2c_bwd_avx2.c"
    , [ "15"; "--cil-t2c"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix15_z_n1c_avx2.c"
    , [ "15"; "--cil-n1c"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix15_z_n1c_bwd_avx2.c"
    , [ "15"; "--cil-n1c"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix17_z_t2c_avx2.c"
    , [ "17"; "--cil-t2c"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix17_z_t2c_bwd_avx2.c"
    , [ "17"; "--cil-t2c"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix17_z_n1c_avx2.c"
    , [ "17"; "--cil-n1c"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix17_z_n1c_bwd_avx2.c"
    , [ "17"; "--cil-n1c"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix19_z_t2c_avx2.c"
    , [ "19"; "--cil-t2c"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix23_z_t2c_avx2.c"
    , [ "23"; "--cil-t2c"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix43_z_t2c_avx2.c"
    , [ "43"; "--cil-t2c"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix47_z_t2c_avx2.c"
    , [ "47"; "--cil-t2c"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix37_z_t2c_avx2.c"
    , [ "37"; "--cil-t2c"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix41_z_t2c_avx2.c"
    , [ "41"; "--cil-t2c"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix29_z_t2c_avx2.c"
    , [ "29"; "--cil-t2c"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix31_z_t2c_avx2.c"
    , [ "31"; "--cil-t2c"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix19_z_t2c_bwd_avx2.c"
    , [ "19"; "--cil-t2c"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix23_z_t2c_bwd_avx2.c"
    , [ "23"; "--cil-t2c"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix43_z_t2c_bwd_avx2.c"
    , [ "43"; "--cil-t2c"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix47_z_t2c_bwd_avx2.c"
    , [ "47"; "--cil-t2c"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix37_z_t2c_bwd_avx2.c"
    , [ "37"; "--cil-t2c"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix41_z_t2c_bwd_avx2.c"
    , [ "41"; "--cil-t2c"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix29_z_t2c_bwd_avx2.c"
    , [ "29"; "--cil-t2c"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix31_z_t2c_bwd_avx2.c"
    , [ "31"; "--cil-t2c"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix19_z_n1c_avx2.c"
    , [ "19"; "--cil-n1c"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix23_z_n1c_avx2.c"
    , [ "23"; "--cil-n1c"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix43_z_n1c_avx2.c"
    , [ "43"; "--cil-n1c"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix47_z_n1c_avx2.c"
    , [ "47"; "--cil-n1c"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix37_z_n1c_avx2.c"
    , [ "37"; "--cil-n1c"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix41_z_n1c_avx2.c"
    , [ "41"; "--cil-n1c"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix29_z_n1c_avx2.c"
    , [ "29"; "--cil-n1c"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix31_z_n1c_avx2.c"
    , [ "31"; "--cil-n1c"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix19_z_n1c_bwd_avx2.c"
    , [ "19"; "--cil-n1c"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix23_z_n1c_bwd_avx2.c"
    , [ "23"; "--cil-n1c"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix43_z_n1c_bwd_avx2.c"
    , [ "43"; "--cil-n1c"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix47_z_n1c_bwd_avx2.c"
    , [ "47"; "--cil-n1c"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix37_z_n1c_bwd_avx2.c"
    , [ "37"; "--cil-n1c"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix41_z_n1c_bwd_avx2.c"
    , [ "41"; "--cil-n1c"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix29_z_n1c_bwd_avx2.c"
    , [ "29"; "--cil-n1c"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix31_z_n1c_bwd_avx2.c"
    , [ "31"; "--cil-n1c"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix21_z_t2c_avx2.c"
    , [ "21"; "--cil-t2c"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix21_z_t2c_bwd_avx2.c"
    , [ "21"; "--cil-t2c"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix21_z_n1c_avx2.c"
    , [ "21"; "--cil-n1c"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix21_z_n1c_bwd_avx2.c"
    , [ "21"; "--cil-n1c"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix25_z_t2c_avx2.c"
    , [ "25"; "--cil-t2c"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix25_z_t2c_bwd_avx2.c"
    , [ "25"; "--cil-t2c"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix25_z_n1c_avx2.c"
    , [ "25"; "--cil-n1c"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix25_z_n1c_bwd_avx2.c"
    , [ "25"; "--cil-n1c"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix27_z_t2c_avx2.c"
    , [ "27"; "--cil-t2c"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix27_z_t2c_bwd_avx2.c"
    , [ "27"; "--cil-t2c"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix27_z_n1c_avx2.c"
    , [ "27"; "--cil-n1c"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix27_z_n1c_bwd_avx2.c"
    , [ "27"; "--cil-n1c"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix32_z_t2c_avx2.c"
    , [ "32"; "--cil-t2c"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix32_z_t2c_bwd_avx2.c"
    , [ "32"; "--cil-t2c"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix64_z_t2c_avx2.c"
    , [ "64"; "--cil-t2c"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix64_z_t2c_bwd_avx2.c"
    , [ "64"; "--cil-t2c"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix3_z_t2cp_avx2.c"
    , [ "3"; "--cil-t2cp"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix5_z_t2cp_avx2.c"
    , [ "5"; "--cil-t2cp"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix7_z_t2cp_avx2.c"
    , [ "7"; "--cil-t2cp"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix9_z_t2cp_avx2.c"
    , [ "9"; "--cil-t2cp"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix11_z_t2cp_avx2.c"
    , [ "11"; "--cil-t2cp"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix13_z_t2cp_avx2.c"
    , [ "13"; "--cil-t2cp"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix15_z_t2cp_avx2.c"
    , [ "15"; "--cil-t2cp"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix17_z_t2cp_avx2.c"
    , [ "17"; "--cil-t2cp"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix19_z_t2cp_avx2.c"
    , [ "19"; "--cil-t2cp"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix23_z_t2cp_avx2.c"
    , [ "23"; "--cil-t2cp"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix43_z_t2cp_avx2.c"
    , [ "43"; "--cil-t2cp"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix47_z_t2cp_avx2.c"
    , [ "47"; "--cil-t2cp"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix37_z_t2cp_avx2.c"
    , [ "37"; "--cil-t2cp"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix41_z_t2cp_avx2.c"
    , [ "41"; "--cil-t2cp"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix29_z_t2cp_avx2.c"
    , [ "29"; "--cil-t2cp"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix31_z_t2cp_avx2.c"
    , [ "31"; "--cil-t2cp"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix21_z_t2cp_avx2.c"
    , [ "21"; "--cil-t2cp"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix25_z_t2cp_avx2.c"
    , [ "25"; "--cil-t2cp"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix27_z_t2cp_avx2.c"
    , [ "27"; "--cil-t2cp"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix4_z_t2cp_avx2.c"
    , [ "4"; "--cil-t2cp"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix8_z_t2cp_avx2.c"
    , [ "8"; "--cil-t2cp"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix16_z_t2cp_avx2.c"
    , [ "16"; "--cil-t2cp"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix3_z_t2cs_avx2.c"
    , [ "3"; "--cil-t2cs"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix5_z_t2cs_avx2.c"
    , [ "5"; "--cil-t2cs"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix7_z_t2cs_avx2.c"
    , [ "7"; "--cil-t2cs"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix9_z_t2cs_avx2.c"
    , [ "9"; "--cil-t2cs"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix11_z_t2cs_avx2.c"
    , [ "11"; "--cil-t2cs"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix13_z_t2cs_avx2.c"
    , [ "13"; "--cil-t2cs"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix15_z_t2cs_avx2.c"
    , [ "15"; "--cil-t2cs"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix17_z_t2cs_avx2.c"
    , [ "17"; "--cil-t2cs"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix19_z_t2cs_avx2.c"
    , [ "19"; "--cil-t2cs"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix23_z_t2cs_avx2.c"
    , [ "23"; "--cil-t2cs"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix43_z_t2cs_avx2.c"
    , [ "43"; "--cil-t2cs"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix47_z_t2cs_avx2.c"
    , [ "47"; "--cil-t2cs"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix37_z_t2cs_avx2.c"
    , [ "37"; "--cil-t2cs"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix41_z_t2cs_avx2.c"
    , [ "41"; "--cil-t2cs"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix29_z_t2cs_avx2.c"
    , [ "29"; "--cil-t2cs"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix31_z_t2cs_avx2.c"
    , [ "31"; "--cil-t2cs"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix21_z_t2cs_avx2.c"
    , [ "21"; "--cil-t2cs"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix25_z_t2cs_avx2.c"
    , [ "25"; "--cil-t2cs"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix27_z_t2cs_avx2.c"
    , [ "27"; "--cil-t2cs"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix4_z_t2cs_avx2.c"
    , [ "4"; "--cil-t2cs"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix8_z_t2cs_avx2.c"
    , [ "8"; "--cil-t2cs"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix16_z_t2cs_avx2.c"
    , [ "16"; "--cil-t2cs"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix3_z_t2csg_avx2.c"
    , [ "3"; "--cil-t2csg"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix5_z_t2csg_avx2.c"
    , [ "5"; "--cil-t2csg"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix7_z_t2csg_avx2.c"
    , [ "7"; "--cil-t2csg"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix9_z_t2csg_avx2.c"
    , [ "9"; "--cil-t2csg"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix11_z_t2csg_avx2.c"
    , [ "11"; "--cil-t2csg"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix13_z_t2csg_avx2.c"
    , [ "13"; "--cil-t2csg"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix15_z_t2csg_avx2.c"
    , [ "15"; "--cil-t2csg"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix17_z_t2csg_avx2.c"
    , [ "17"; "--cil-t2csg"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix19_z_t2csg_avx2.c"
    , [ "19"; "--cil-t2csg"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix23_z_t2csg_avx2.c"
    , [ "23"; "--cil-t2csg"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix43_z_t2csg_avx2.c"
    , [ "43"; "--cil-t2csg"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix47_z_t2csg_avx2.c"
    , [ "47"; "--cil-t2csg"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix37_z_t2csg_avx2.c"
    , [ "37"; "--cil-t2csg"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix41_z_t2csg_avx2.c"
    , [ "41"; "--cil-t2csg"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix29_z_t2csg_avx2.c"
    , [ "29"; "--cil-t2csg"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix31_z_t2csg_avx2.c"
    , [ "31"; "--cil-t2csg"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix21_z_t2csg_avx2.c"
    , [ "21"; "--cil-t2csg"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix25_z_t2csg_avx2.c"
    , [ "25"; "--cil-t2csg"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix27_z_t2csg_avx2.c"
    , [ "27"; "--cil-t2csg"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix4_z_t2csg_avx2.c"
    , [ "4"; "--cil-t2csg"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix8_z_t2csg_avx2.c"
    , [ "8"; "--cil-t2csg"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix16_z_t2csg_avx2.c"
    , [ "16"; "--cil-t2csg"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix3_z_t2csgn_avx2.c"
    , [ "3"; "--cil-t2csgn"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix5_z_t2csgn_avx2.c"
    , [ "5"; "--cil-t2csgn"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix7_z_t2csgn_avx2.c"
    , [ "7"; "--cil-t2csgn"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix9_z_t2csgn_avx2.c"
    , [ "9"; "--cil-t2csgn"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix11_z_t2csgn_avx2.c"
    , [ "11"; "--cil-t2csgn"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix13_z_t2csgn_avx2.c"
    , [ "13"; "--cil-t2csgn"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix15_z_t2csgn_avx2.c"
    , [ "15"; "--cil-t2csgn"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix17_z_t2csgn_avx2.c"
    , [ "17"; "--cil-t2csgn"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix19_z_t2csgn_avx2.c"
    , [ "19"; "--cil-t2csgn"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix23_z_t2csgn_avx2.c"
    , [ "23"; "--cil-t2csgn"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix43_z_t2csgn_avx2.c"
    , [ "43"; "--cil-t2csgn"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix47_z_t2csgn_avx2.c"
    , [ "47"; "--cil-t2csgn"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix37_z_t2csgn_avx2.c"
    , [ "37"; "--cil-t2csgn"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix41_z_t2csgn_avx2.c"
    , [ "41"; "--cil-t2csgn"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix29_z_t2csgn_avx2.c"
    , [ "29"; "--cil-t2csgn"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix31_z_t2csgn_avx2.c"
    , [ "31"; "--cil-t2csgn"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix21_z_t2csgn_avx2.c"
    , [ "21"; "--cil-t2csgn"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix25_z_t2csgn_avx2.c"
    , [ "25"; "--cil-t2csgn"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix27_z_t2csgn_avx2.c"
    , [ "27"; "--cil-t2csgn"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix4_z_t2csgn_avx2.c"
    , [ "4"; "--cil-t2csgn"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix8_z_t2csgn_avx2.c"
    , [ "8"; "--cil-t2csgn"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix16_z_t2csgn_avx2.c"
    , [ "16"; "--cil-t2csgn"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix3_z_t2csg_bwd_avx2.c"
    , [ "3"; "--cil-t2csg"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix3_z_t2csgn_bwd_avx2.c"
    , [ "3"; "--cil-t2csgn"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix5_z_t2csg_bwd_avx2.c"
    , [ "5"; "--cil-t2csg"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix5_z_t2csgn_bwd_avx2.c"
    , [ "5"; "--cil-t2csgn"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix7_z_t2csg_bwd_avx2.c"
    , [ "7"; "--cil-t2csg"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix7_z_t2csgn_bwd_avx2.c"
    , [ "7"; "--cil-t2csgn"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix9_z_t2csg_bwd_avx2.c"
    , [ "9"; "--cil-t2csg"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix9_z_t2csgn_bwd_avx2.c"
    , [ "9"; "--cil-t2csgn"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix11_z_t2csg_bwd_avx2.c"
    , [ "11"; "--cil-t2csg"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix11_z_t2csgn_bwd_avx2.c"
    , [ "11"; "--cil-t2csgn"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix13_z_t2csg_bwd_avx2.c"
    , [ "13"; "--cil-t2csg"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix13_z_t2csgn_bwd_avx2.c"
    , [ "13"; "--cil-t2csgn"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix15_z_t2csg_bwd_avx2.c"
    , [ "15"; "--cil-t2csg"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix15_z_t2csgn_bwd_avx2.c"
    , [ "15"; "--cil-t2csgn"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix17_z_t2csg_bwd_avx2.c"
    , [ "17"; "--cil-t2csg"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix17_z_t2csgn_bwd_avx2.c"
    , [ "17"; "--cil-t2csgn"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix19_z_t2csg_bwd_avx2.c"
    , [ "19"; "--cil-t2csg"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix23_z_t2csg_bwd_avx2.c"
    , [ "23"; "--cil-t2csg"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix43_z_t2csg_bwd_avx2.c"
    , [ "43"; "--cil-t2csg"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix47_z_t2csg_bwd_avx2.c"
    , [ "47"; "--cil-t2csg"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix37_z_t2csg_bwd_avx2.c"
    , [ "37"; "--cil-t2csg"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix41_z_t2csg_bwd_avx2.c"
    , [ "41"; "--cil-t2csg"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix29_z_t2csg_bwd_avx2.c"
    , [ "29"; "--cil-t2csg"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix31_z_t2csg_bwd_avx2.c"
    , [ "31"; "--cil-t2csg"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix19_z_t2csgn_bwd_avx2.c"
    , [ "19"; "--cil-t2csgn"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix23_z_t2csgn_bwd_avx2.c"
    , [ "23"; "--cil-t2csgn"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix43_z_t2csgn_bwd_avx2.c"
    , [ "43"; "--cil-t2csgn"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix47_z_t2csgn_bwd_avx2.c"
    , [ "47"; "--cil-t2csgn"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix37_z_t2csgn_bwd_avx2.c"
    , [ "37"; "--cil-t2csgn"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix41_z_t2csgn_bwd_avx2.c"
    , [ "41"; "--cil-t2csgn"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix29_z_t2csgn_bwd_avx2.c"
    , [ "29"; "--cil-t2csgn"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix31_z_t2csgn_bwd_avx2.c"
    , [ "31"; "--cil-t2csgn"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix21_z_t2csg_bwd_avx2.c"
    , [ "21"; "--cil-t2csg"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix21_z_t2csgn_bwd_avx2.c"
    , [ "21"; "--cil-t2csgn"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix25_z_t2csg_bwd_avx2.c"
    , [ "25"; "--cil-t2csg"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix25_z_t2csgn_bwd_avx2.c"
    , [ "25"; "--cil-t2csgn"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix27_z_t2csg_bwd_avx2.c"
    , [ "27"; "--cil-t2csg"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix27_z_t2csgn_bwd_avx2.c"
    , [ "27"; "--cil-t2csgn"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix4_z_t2csg_bwd_avx2.c"
    , [ "4"; "--cil-t2csg"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix4_z_t2csgn_bwd_avx2.c"
    , [ "4"; "--cil-t2csgn"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix8_z_t2csg_bwd_avx2.c"
    , [ "8"; "--cil-t2csg"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix8_z_t2csgn_bwd_avx2.c"
    , [ "8"; "--cil-t2csgn"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix16_z_t2csg_bwd_avx2.c"
    , [ "16"; "--cil-t2csg"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix16_z_t2csgn_bwd_avx2.c"
    , [ "16"; "--cil-t2csgn"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix3_z_t2cp_bwd_avx2.c"
    , [ "3"; "--cil-t2cp"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix3_z_t2csgt_bwd_avx2.c"
    , [ "3"; "--cil-t2csgt"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix3_z_t2csgnt_bwd_avx2.c"
    , [ "3"; "--cil-t2csgnt"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix5_z_t2cp_bwd_avx2.c"
    , [ "5"; "--cil-t2cp"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix5_z_t2csgt_bwd_avx2.c"
    , [ "5"; "--cil-t2csgt"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix5_z_t2csgnt_bwd_avx2.c"
    , [ "5"; "--cil-t2csgnt"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix7_z_t2cp_bwd_avx2.c"
    , [ "7"; "--cil-t2cp"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix7_z_t2csgt_bwd_avx2.c"
    , [ "7"; "--cil-t2csgt"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix7_z_t2csgnt_bwd_avx2.c"
    , [ "7"; "--cil-t2csgnt"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix9_z_t2cp_bwd_avx2.c"
    , [ "9"; "--cil-t2cp"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix9_z_t2csgt_bwd_avx2.c"
    , [ "9"; "--cil-t2csgt"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix9_z_t2csgnt_bwd_avx2.c"
    , [ "9"; "--cil-t2csgnt"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix11_z_t2cp_bwd_avx2.c"
    , [ "11"; "--cil-t2cp"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix11_z_t2csgt_bwd_avx2.c"
    , [ "11"; "--cil-t2csgt"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix11_z_t2csgnt_bwd_avx2.c"
    , [ "11"; "--cil-t2csgnt"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix13_z_t2cp_bwd_avx2.c"
    , [ "13"; "--cil-t2cp"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix13_z_t2csgt_bwd_avx2.c"
    , [ "13"; "--cil-t2csgt"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix13_z_t2csgnt_bwd_avx2.c"
    , [ "13"; "--cil-t2csgnt"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix15_z_t2cp_bwd_avx2.c"
    , [ "15"; "--cil-t2cp"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix15_z_t2csgt_bwd_avx2.c"
    , [ "15"; "--cil-t2csgt"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix15_z_t2csgnt_bwd_avx2.c"
    , [ "15"; "--cil-t2csgnt"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix17_z_t2cp_bwd_avx2.c"
    , [ "17"; "--cil-t2cp"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix17_z_t2csgt_bwd_avx2.c"
    , [ "17"; "--cil-t2csgt"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix17_z_t2csgnt_bwd_avx2.c"
    , [ "17"; "--cil-t2csgnt"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix19_z_t2cp_bwd_avx2.c"
    , [ "19"; "--cil-t2cp"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix23_z_t2cp_bwd_avx2.c"
    , [ "23"; "--cil-t2cp"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix43_z_t2cp_bwd_avx2.c"
    , [ "43"; "--cil-t2cp"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix47_z_t2cp_bwd_avx2.c"
    , [ "47"; "--cil-t2cp"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix37_z_t2cp_bwd_avx2.c"
    , [ "37"; "--cil-t2cp"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix41_z_t2cp_bwd_avx2.c"
    , [ "41"; "--cil-t2cp"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix29_z_t2cp_bwd_avx2.c"
    , [ "29"; "--cil-t2cp"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix31_z_t2cp_bwd_avx2.c"
    , [ "31"; "--cil-t2cp"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix19_z_t2csgt_bwd_avx2.c"
    , [ "19"; "--cil-t2csgt"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix23_z_t2csgt_bwd_avx2.c"
    , [ "23"; "--cil-t2csgt"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix43_z_t2csgt_bwd_avx2.c"
    , [ "43"; "--cil-t2csgt"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix47_z_t2csgt_bwd_avx2.c"
    , [ "47"; "--cil-t2csgt"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix37_z_t2csgt_bwd_avx2.c"
    , [ "37"; "--cil-t2csgt"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix41_z_t2csgt_bwd_avx2.c"
    , [ "41"; "--cil-t2csgt"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix29_z_t2csgt_bwd_avx2.c"
    , [ "29"; "--cil-t2csgt"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix31_z_t2csgt_bwd_avx2.c"
    , [ "31"; "--cil-t2csgt"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix19_z_t2csgnt_bwd_avx2.c"
    , [ "19"; "--cil-t2csgnt"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix23_z_t2csgnt_bwd_avx2.c"
    , [ "23"; "--cil-t2csgnt"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix43_z_t2csgnt_bwd_avx2.c"
    , [ "43"; "--cil-t2csgnt"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix47_z_t2csgnt_bwd_avx2.c"
    , [ "47"; "--cil-t2csgnt"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix37_z_t2csgnt_bwd_avx2.c"
    , [ "37"; "--cil-t2csgnt"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix41_z_t2csgnt_bwd_avx2.c"
    , [ "41"; "--cil-t2csgnt"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix29_z_t2csgnt_bwd_avx2.c"
    , [ "29"; "--cil-t2csgnt"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix31_z_t2csgnt_bwd_avx2.c"
    , [ "31"; "--cil-t2csgnt"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix21_z_t2cp_bwd_avx2.c"
    , [ "21"; "--cil-t2cp"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix21_z_t2csgt_bwd_avx2.c"
    , [ "21"; "--cil-t2csgt"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix21_z_t2csgnt_bwd_avx2.c"
    , [ "21"; "--cil-t2csgnt"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix25_z_t2cp_bwd_avx2.c"
    , [ "25"; "--cil-t2cp"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix25_z_t2csgt_bwd_avx2.c"
    , [ "25"; "--cil-t2csgt"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix25_z_t2csgnt_bwd_avx2.c"
    , [ "25"; "--cil-t2csgnt"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix27_z_t2cp_bwd_avx2.c"
    , [ "27"; "--cil-t2cp"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix27_z_t2csgt_bwd_avx2.c"
    , [ "27"; "--cil-t2csgt"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix27_z_t2csgnt_bwd_avx2.c"
    , [ "27"; "--cil-t2csgnt"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix4_z_t2cp_bwd_avx2.c"
    , [ "4"; "--cil-t2cp"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix4_z_t2csgt_bwd_avx2.c"
    , [ "4"; "--cil-t2csgt"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix4_z_t2csgnt_bwd_avx2.c"
    , [ "4"; "--cil-t2csgnt"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix8_z_t2cp_bwd_avx2.c"
    , [ "8"; "--cil-t2cp"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix8_z_t2csgt_bwd_avx2.c"
    , [ "8"; "--cil-t2csgt"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix8_z_t2csgnt_bwd_avx2.c"
    , [ "8"; "--cil-t2csgnt"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix16_z_t2cp_bwd_avx2.c"
    , [ "16"; "--cil-t2cp"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix16_z_t2csgt_bwd_avx2.c"
    , [ "16"; "--cil-t2csgt"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix16_z_t2csgnt_bwd_avx2.c"
    , [ "16"; "--cil-t2csgnt"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix16_z_t2cb44_avx2.c"
    , [ "16"; "--cil-t2c"; "--cil-blocked"; "--cil-split"; "4.4"; "--cil-form-tag"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix16_z_t2cb44_bwd_avx2.c"
    , [ "16"; "--cil-t2c"; "--cil-blocked"; "--cil-split"; "4.4"; "--cil-form-tag"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix32_z_t2cb48_avx2.c"
    , [ "32"; "--cil-t2c"; "--cil-blocked"; "--cil-split"; "4.8"; "--cil-form-tag"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix32_z_t2cb48_bwd_avx2.c"
    , [ "32"; "--cil-t2c"; "--cil-blocked"; "--cil-split"; "4.8"; "--cil-form-tag"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix32_z_t2cb84_avx2.c"
    , [ "32"; "--cil-t2c"; "--cil-blocked"; "--cil-split"; "8.4"; "--cil-form-tag"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix32_z_t2cb84_bwd_avx2.c"
    , [ "32"; "--cil-t2c"; "--cil-blocked"; "--cil-split"; "8.4"; "--cil-form-tag"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix64_z_t2cb88_avx2.c"
    , [ "64"; "--cil-t2c"; "--cil-blocked"; "--cil-split"; "8.8"; "--cil-form-tag"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix64_z_t2cb88_bwd_avx2.c"
    , [ "64"; "--cil-t2c"; "--cil-blocked"; "--cil-split"; "8.8"; "--cil-form-tag"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix32_z_n1cb48_avx2.c"
    , [ "32"; "--cil-n1c"; "--cil-blocked"; "--cil-split"; "4.8"; "--cil-form-tag"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix32_z_n1cb48_bwd_avx2.c"
    , [ "32"; "--cil-n1c"; "--cil-blocked"; "--cil-split"; "4.8"; "--cil-form-tag"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix32_z_n1cb84_avx2.c"
    , [ "32"; "--cil-n1c"; "--cil-blocked"; "--cil-split"; "8.4"; "--cil-form-tag"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix32_z_n1cb84_bwd_avx2.c"
    , [ "32"; "--cil-n1c"; "--cil-blocked"; "--cil-split"; "8.4"; "--cil-form-tag"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix64_z_n1cb88_avx2.c"
    , [ "64"; "--cil-n1c"; "--cil-blocked"; "--cil-split"; "8.8"; "--cil-form-tag"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix64_z_n1cb88_bwd_avx2.c"
    , [ "64"; "--cil-n1c"; "--cil-blocked"; "--cil-split"; "8.8"; "--cil-form-tag"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
    (* the 128 column leaf (2026-10-02): the whole column pass of an N1 = 128
       plane in ONE kernel, natural in place by construction (no second
       plane); two-pass 8.16 and three-pass 4.4.8, forward only -- the 2D
       real r2c column plan's leaf forms, raced per cell (b816 for few
       columns, b448 for many) *)
  ; ( "radix128_z_n1cb816_avx2.c"
    , [ "128"; "--cil-n1c"; "--cil-blocked"; "--cil-split"; "8.16"; "--cil-form-tag"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix128_z_n1cb448_avx2.c"
    , [ "128"; "--cil-n1c"; "--cil-blocked"; "--cil-split3"; "4.4.8"; "--cil-form-tag"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  (* their backward (2026-10-05): the 2D real c2r column plan's one-kernel leaves *)
  ; ( "radix128_z_n1cb816_bwd_avx2.c"
    , [ "128"; "--cil-n1c"; "--cil-blocked"; "--cil-split"; "8.16"; "--cil-form-tag"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix128_z_n1cb448_bwd_avx2.c"
    , [ "128"; "--cil-n1c"; "--cil-blocked"; "--cil-split3"; "4.4.8"; "--cil-form-tag"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
    (* the n1c column leaf's HALF-STORE twins (--cil-st128, the "h" form,
       2026-10-01): the plane stores as two 128-bit halves -- the odd CCE pitch
       of the 2D real column pass; forward only (the 2D real r2c leaf), raced
       per leaf stage; zil/avx2/shared/col/half *)
  ; ( "radix2_z_n1ch_avx2.c"
    , [ "2"; "--cil-n1c"; "--cil-st128"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix3_z_n1ch_avx2.c"
    , [ "3"; "--cil-n1c"; "--cil-st128"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix4_z_n1ch_avx2.c"
    , [ "4"; "--cil-n1c"; "--cil-st128"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix5_z_n1ch_avx2.c"
    , [ "5"; "--cil-n1c"; "--cil-st128"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix6_z_n1ch_avx2.c"
    , [ "6"; "--cil-n1c"; "--cil-st128"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix7_z_n1ch_avx2.c"
    , [ "7"; "--cil-n1c"; "--cil-st128"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix8_z_n1ch_avx2.c"
    , [ "8"; "--cil-n1c"; "--cil-st128"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix9_z_n1ch_avx2.c"
    , [ "9"; "--cil-n1c"; "--cil-st128"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix10_z_n1ch_avx2.c"
    , [ "10"; "--cil-n1c"; "--cil-st128"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix11_z_n1ch_avx2.c"
    , [ "11"; "--cil-n1c"; "--cil-st128"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix12_z_n1ch_avx2.c"
    , [ "12"; "--cil-n1c"; "--cil-st128"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix13_z_n1ch_avx2.c"
    , [ "13"; "--cil-n1c"; "--cil-st128"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix14_z_n1ch_avx2.c"
    , [ "14"; "--cil-n1c"; "--cil-st128"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix15_z_n1ch_avx2.c"
    , [ "15"; "--cil-n1c"; "--cil-st128"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix16_z_n1ch_avx2.c"
    , [ "16"; "--cil-n1c"; "--cil-st128"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix17_z_n1ch_avx2.c"
    , [ "17"; "--cil-n1c"; "--cil-st128"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix19_z_n1ch_avx2.c"
    , [ "19"; "--cil-n1c"; "--cil-st128"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix21_z_n1ch_avx2.c"
    , [ "21"; "--cil-n1c"; "--cil-st128"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix22_z_n1ch_avx2.c"
    , [ "22"; "--cil-n1c"; "--cil-st128"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix23_z_n1ch_avx2.c"
    , [ "23"; "--cil-n1c"; "--cil-st128"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix25_z_n1ch_avx2.c"
    , [ "25"; "--cil-n1c"; "--cil-st128"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix26_z_n1ch_avx2.c"
    , [ "26"; "--cil-n1c"; "--cil-st128"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix27_z_n1ch_avx2.c"
    , [ "27"; "--cil-n1c"; "--cil-st128"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix29_z_n1ch_avx2.c"
    , [ "29"; "--cil-n1c"; "--cil-st128"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix31_z_n1ch_avx2.c"
    , [ "31"; "--cil-n1c"; "--cil-st128"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix37_z_n1ch_avx2.c"
    , [ "37"; "--cil-n1c"; "--cil-st128"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix41_z_n1ch_avx2.c"
    , [ "41"; "--cil-n1c"; "--cil-st128"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix43_z_n1ch_avx2.c"
    , [ "43"; "--cil-n1c"; "--cil-st128"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix47_z_n1ch_avx2.c"
    , [ "47"; "--cil-n1c"; "--cil-st128"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix32_z_n1cb48h_avx2.c"
    , [ "32"; "--cil-n1c"; "--cil-blocked"; "--cil-split"; "4.8"; "--cil-form-tag"; "--cil-st128"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix32_z_n1cb84h_avx2.c"
    , [ "32"; "--cil-n1c"; "--cil-blocked"; "--cil-split"; "8.4"; "--cil-form-tag"; "--cil-st128"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix64_z_n1cb88h_avx2.c"
    , [ "64"; "--cil-n1c"; "--cil-blocked"; "--cil-split"; "8.8"; "--cil-form-tag"; "--cil-st128"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix10_z_n1ccs_avx2.c"
    , [ "10"; "--cil-n1ccs"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix10_z_n1ccs_bwd_avx2.c"
    , [ "10"; "--cil-n1ccs"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix14_z_n1ccs_avx2.c"
    , [ "14"; "--cil-n1ccs"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix14_z_n1ccs_bwd_avx2.c"
    , [ "14"; "--cil-n1ccs"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix22_z_n1ccs_avx2.c"
    , [ "22"; "--cil-n1ccs"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix22_z_n1ccs_bwd_avx2.c"
    , [ "22"; "--cil-n1ccs"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix26_z_n1ccs_avx2.c"
    , [ "26"; "--cil-n1ccs"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix26_z_n1ccs_bwd_avx2.c"
    , [ "26"; "--cil-n1ccs"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix12_z_n1ccs_avx2.c"
    , [ "12"; "--cil-n1ccs"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix12_z_n1ccs_bwd_avx2.c"
    , [ "12"; "--cil-n1ccs"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix2_z_n1ccs_avx2.c"
    , [ "2"; "--cil-n1ccs"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix2_z_n1ccs_bwd_avx2.c"
    , [ "2"; "--cil-n1ccs"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix6_z_n1ccs_avx2.c"
    , [ "6"; "--cil-n1ccs"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix6_z_n1ccs_bwd_avx2.c"
    , [ "6"; "--cil-n1ccs"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix4_z_n1ccs_avx2.c"
    , [ "4"; "--cil-n1ccs"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix4_z_n1ccs_bwd_avx2.c"
    , [ "4"; "--cil-n1ccs"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix8_z_n1ccs_avx2.c"
    , [ "8"; "--cil-n1ccs"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix8_z_n1ccs_bwd_avx2.c"
    , [ "8"; "--cil-n1ccs"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix16_z_n1ccs_avx2.c"
    , [ "16"; "--cil-n1ccs"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix16_z_n1ccs_bwd_avx2.c"
    , [ "16"; "--cil-n1ccs"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix32_z_n1ccs_avx2.c"
    , [ "32"; "--cil-n1ccs"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix32_z_n1ccs_bwd_avx2.c"
    , [ "32"; "--cil-n1ccs"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix64_z_n1ccs_avx2.c"
    , [ "64"; "--cil-n1ccs"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix64_z_n1ccs_bwd_avx2.c"
    , [ "64"; "--cil-n1ccs"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix3_z_n1ccs_avx2.c"
    , [ "3"; "--cil-n1ccs"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix3_z_n1ccs_bwd_avx2.c"
    , [ "3"; "--cil-n1ccs"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix5_z_n1ccs_avx2.c"
    , [ "5"; "--cil-n1ccs"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix5_z_n1ccs_bwd_avx2.c"
    , [ "5"; "--cil-n1ccs"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix7_z_n1ccs_avx2.c"
    , [ "7"; "--cil-n1ccs"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix7_z_n1ccs_bwd_avx2.c"
    , [ "7"; "--cil-n1ccs"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix9_z_n1ccs_avx2.c"
    , [ "9"; "--cil-n1ccs"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix9_z_n1ccs_bwd_avx2.c"
    , [ "9"; "--cil-n1ccs"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix11_z_n1ccs_avx2.c"
    , [ "11"; "--cil-n1ccs"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix11_z_n1ccs_bwd_avx2.c"
    , [ "11"; "--cil-n1ccs"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix13_z_n1ccs_avx2.c"
    , [ "13"; "--cil-n1ccs"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix13_z_n1ccs_bwd_avx2.c"
    , [ "13"; "--cil-n1ccs"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix15_z_n1ccs_avx2.c"
    , [ "15"; "--cil-n1ccs"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix15_z_n1ccs_bwd_avx2.c"
    , [ "15"; "--cil-n1ccs"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix17_z_n1ccs_avx2.c"
    , [ "17"; "--cil-n1ccs"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix17_z_n1ccs_bwd_avx2.c"
    , [ "17"; "--cil-n1ccs"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix19_z_n1ccs_avx2.c"
    , [ "19"; "--cil-n1ccs"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix23_z_n1ccs_avx2.c"
    , [ "23"; "--cil-n1ccs"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix43_z_n1ccs_avx2.c"
    , [ "43"; "--cil-n1ccs"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix47_z_n1ccs_avx2.c"
    , [ "47"; "--cil-n1ccs"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix37_z_n1ccs_avx2.c"
    , [ "37"; "--cil-n1ccs"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix41_z_n1ccs_avx2.c"
    , [ "41"; "--cil-n1ccs"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix29_z_n1ccs_avx2.c"
    , [ "29"; "--cil-n1ccs"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix31_z_n1ccs_avx2.c"
    , [ "31"; "--cil-n1ccs"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix19_z_n1ccs_bwd_avx2.c"
    , [ "19"; "--cil-n1ccs"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix23_z_n1ccs_bwd_avx2.c"
    , [ "23"; "--cil-n1ccs"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix43_z_n1ccs_bwd_avx2.c"
    , [ "43"; "--cil-n1ccs"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix47_z_n1ccs_bwd_avx2.c"
    , [ "47"; "--cil-n1ccs"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix37_z_n1ccs_bwd_avx2.c"
    , [ "37"; "--cil-n1ccs"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix41_z_n1ccs_bwd_avx2.c"
    , [ "41"; "--cil-n1ccs"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix29_z_n1ccs_bwd_avx2.c"
    , [ "29"; "--cil-n1ccs"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix31_z_n1ccs_bwd_avx2.c"
    , [ "31"; "--cil-n1ccs"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix21_z_n1ccs_avx2.c"
    , [ "21"; "--cil-n1ccs"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix21_z_n1ccs_bwd_avx2.c"
    , [ "21"; "--cil-n1ccs"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix25_z_n1ccs_avx2.c"
    , [ "25"; "--cil-n1ccs"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix25_z_n1ccs_bwd_avx2.c"
    , [ "25"; "--cil-n1ccs"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix27_z_n1ccs_avx2.c"
    , [ "27"; "--cil-n1ccs"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix27_z_n1ccs_bwd_avx2.c"
    , [ "27"; "--cil-n1ccs"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
    (* THE ROW FAMILY (2026-11-08, rows/row and rows/rowswap): the batched row kernels that beat n1ccs in the
       pinned race of 2026-11-08 at 256 bits, one winner per radix; n1ccs stays where it won (2 5 6 7 8 10 14
       26 and the odd lengths 17..47) and everywhere for the turn route's turned stores. row at 4 12 16; the
       stage-1 split forms rowb44 at 32 and rowb48 at 64; rowswap at 3 9 11 13 15 22. *)
  ; ( "radix4_z_row_avx2.c", [ "4"; "--cil-row"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix4_z_row_bwd_avx2.c", [ "4"; "--cil-row"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix12_z_row_avx2.c", [ "12"; "--cil-row"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix12_z_row_bwd_avx2.c", [ "12"; "--cil-row"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix16_z_row_avx2.c", [ "16"; "--cil-row"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix16_z_row_bwd_avx2.c", [ "16"; "--cil-row"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix32_z_rowb44_avx2.c", [ "32"; "--cil-row"; "--cil-split"; "4.4"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix32_z_rowb44_bwd_avx2.c", [ "32"; "--cil-row"; "--cil-split"; "4.4"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix64_z_rowb48_avx2.c", [ "64"; "--cil-row"; "--cil-split"; "4.8"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix64_z_rowb48_bwd_avx2.c", [ "64"; "--cil-row"; "--cil-split"; "4.8"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix3_z_rowswap_avx2.c", [ "3"; "--cil-rowswap"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix3_z_rowswap_bwd_avx2.c", [ "3"; "--cil-rowswap"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix9_z_rowswap_avx2.c", [ "9"; "--cil-rowswap"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix9_z_rowswap_bwd_avx2.c", [ "9"; "--cil-rowswap"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix11_z_rowswap_avx2.c", [ "11"; "--cil-rowswap"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix11_z_rowswap_bwd_avx2.c", [ "11"; "--cil-rowswap"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix13_z_rowswap_avx2.c", [ "13"; "--cil-rowswap"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix13_z_rowswap_bwd_avx2.c", [ "13"; "--cil-rowswap"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix15_z_rowswap_avx2.c", [ "15"; "--cil-rowswap"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix15_z_rowswap_bwd_avx2.c", [ "15"; "--cil-rowswap"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix22_z_rowswap_avx2.c", [ "22"; "--cil-rowswap"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix22_z_rowswap_bwd_avx2.c", [ "22"; "--cil-rowswap"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
    (* THE TURNED TWINS (--cil-turnst on the row kinds, 2026-11-08) are GENERATE-ABLE BUT NOT SHIPPED: the 2D turn
       route's row kernels, storing point-major into its scratch (rowswapt, rowb44t, rowb48t). In the turn role
       they beat n1ccs per row (rowswapt 1.07x..1.59x at 2 3 5 7 8 9 10 11 13 14 15, rowb44t 2.2x, rowb48t 2.0x),
       but no PLANE got faster: the planner takes the turn route only for planes 2..16 points wide, where the row
       pass is a small share of the time. Owner 11-08: removed from the tree, kept in the generator, to be raced
       again on AVX-512 and on the odd-sized planes. A twin's cell is
         ( "radixR_z_rowswapt_avx2.c", [ "R"; "--cil-rowswap"; "--cil-turnst"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
       and emit_il_registry's VFFT_IL_ROWT_CHOICE picks it up; with none shipped that choice is n1ccs at every radix. *)
  ; ( "radix4_z_n1tr_avx2.c"
    , [ "4"; "--cil-n1t"; "--cil-rowloop"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix4_z_t2r_avx2.c"
    , [ "4"; "--cil-t2"; "--cil-rowloop"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix4_z_t2tr_bwd_avx2.c"
    , [ "4"; "--cil-t2"; "--cil-rowloop"; "--cil-turnst"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix4_z_n1r_bwd_avx2.c"
    , [ "4"; "--cil-n1"; "--cil-rowloop"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix8_z_n1tr_avx2.c"
    , [ "8"; "--cil-n1t"; "--cil-rowloop"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix8_z_t2r_avx2.c"
    , [ "8"; "--cil-t2"; "--cil-rowloop"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix8_z_t2tr_bwd_avx2.c"
    , [ "8"; "--cil-t2"; "--cil-rowloop"; "--cil-turnst"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix8_z_n1r_bwd_avx2.c"
    , [ "8"; "--cil-n1"; "--cil-rowloop"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix16_z_n1tr_avx2.c"
    , [ "16"; "--cil-n1t"; "--cil-rowloop"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix16_z_t2r_avx2.c"
    , [ "16"; "--cil-t2"; "--cil-rowloop"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix16_z_t2tr_bwd_avx2.c"
    , [ "16"; "--cil-t2"; "--cil-rowloop"; "--cil-turnst"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix16_z_n1r_bwd_avx2.c"
    , [ "16"; "--cil-n1"; "--cil-rowloop"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix8_z_n1trtan_avx2.c"
    , [ "8"; "--cil-n1t"; "--cil-tangent"; "--cil-rowloop"; "--cil-form-tag"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix8_z_t2rtan_avx2.c"
    , [ "8"; "--cil-t2"; "--cil-tangent"; "--cil-rowloop"; "--cil-form-tag"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix8_z_t2trtan_bwd_avx2.c"
    , [ "8"; "--cil-t2"; "--cil-tangent"; "--cil-rowloop"; "--cil-turnst"; "--cil-form-tag"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix8_z_n1rtan_bwd_avx2.c"
    , [ "8"; "--cil-n1"; "--cil-tangent"; "--cil-rowloop"; "--cil-form-tag"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix16_z_n1trtan_avx2.c"
    , [ "16"; "--cil-n1t"; "--cil-tangent"; "--cil-rowloop"; "--cil-form-tag"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix16_z_t2rtan_avx2.c"
    , [ "16"; "--cil-t2"; "--cil-tangent"; "--cil-rowloop"; "--cil-form-tag"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix16_z_t2trtan_bwd_avx2.c"
    , [ "16"; "--cil-t2"; "--cil-tangent"; "--cil-rowloop"; "--cil-turnst"; "--cil-form-tag"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix16_z_n1rtan_bwd_avx2.c"
    , [ "16"; "--cil-n1"; "--cil-tangent"; "--cil-rowloop"; "--cil-form-tag"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
    (* the interleaved REAL pair's top stage t2h (real_il.ml, 2026-09-29):
       fwd + bwd at the even pair radices; zil/avx2/real/pair *)
  ; ( "radix4_z_t2h_avx2.c"
    , [ "4"; "--cil-t2h"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix4_z_t2h_bwd_avx2.c"
    , [ "4"; "--cil-t2h"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix6_z_t2h_avx2.c"
    , [ "6"; "--cil-t2h"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix6_z_t2h_bwd_avx2.c"
    , [ "6"; "--cil-t2h"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix8_z_t2h_avx2.c"
    , [ "8"; "--cil-t2h"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix8_z_t2h_bwd_avx2.c"
    , [ "8"; "--cil-t2h"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix10_z_t2h_avx2.c"
    , [ "10"; "--cil-t2h"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix10_z_t2h_bwd_avx2.c"
    , [ "10"; "--cil-t2h"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix12_z_t2h_avx2.c"
    , [ "12"; "--cil-t2h"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix12_z_t2h_bwd_avx2.c"
    , [ "12"; "--cil-t2h"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix16_z_t2h_avx2.c"
    , [ "16"; "--cil-t2h"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix16_z_t2h_bwd_avx2.c"
    , [ "16"; "--cil-t2h"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix32_z_t2h_avx2.c"
    , [ "32"; "--cil-t2h"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix32_z_t2h_bwd_avx2.c"
    , [ "32"; "--cil-t2h"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix64_z_t2h_avx2.c"
    , [ "64"; "--cil-t2h"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix64_z_t2h_bwd_avx2.c"
    , [ "64"; "--cil-t2h"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
    (* form B: the real leaf r2z and its top t2m, fwd + bwd *)
  ; ( "radix4_z_t2m_avx2.c"
    , [ "4"; "--cil-t2m"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix4_z_t2m_bwd_avx2.c"
    , [ "4"; "--cil-t2m"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix6_z_t2m_avx2.c"
    , [ "6"; "--cil-t2m"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix6_z_t2m_bwd_avx2.c"
    , [ "6"; "--cil-t2m"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix8_z_t2m_avx2.c"
    , [ "8"; "--cil-t2m"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix8_z_t2m_bwd_avx2.c"
    , [ "8"; "--cil-t2m"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix10_z_t2m_avx2.c"
    , [ "10"; "--cil-t2m"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix10_z_t2m_bwd_avx2.c"
    , [ "10"; "--cil-t2m"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix12_z_t2m_avx2.c"
    , [ "12"; "--cil-t2m"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix12_z_t2m_bwd_avx2.c"
    , [ "12"; "--cil-t2m"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix16_z_t2m_avx2.c"
    , [ "16"; "--cil-t2m"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix16_z_t2m_bwd_avx2.c"
    , [ "16"; "--cil-t2m"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix32_z_t2m_avx2.c"
    , [ "32"; "--cil-t2m"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix32_z_t2m_bwd_avx2.c"
    , [ "32"; "--cil-t2m"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix64_z_t2m_avx2.c"
    , [ "64"; "--cil-t2m"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix64_z_t2m_bwd_avx2.c"
    , [ "64"; "--cil-t2m"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix4_z_r2z_avx2.c"
    , [ "4"; "--cil-r2z"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix4_z_r2z_bwd_avx2.c"
    , [ "4"; "--cil-r2z"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix6_z_r2z_avx2.c"
    , [ "6"; "--cil-r2z"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix6_z_r2z_bwd_avx2.c"
    , [ "6"; "--cil-r2z"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix8_z_r2z_avx2.c"
    , [ "8"; "--cil-r2z"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix8_z_r2z_bwd_avx2.c"
    , [ "8"; "--cil-r2z"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix10_z_r2z_avx2.c"
    , [ "10"; "--cil-r2z"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix10_z_r2z_bwd_avx2.c"
    , [ "10"; "--cil-r2z"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix12_z_r2z_avx2.c"
    , [ "12"; "--cil-r2z"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix12_z_r2z_bwd_avx2.c"
    , [ "12"; "--cil-r2z"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix16_z_r2z_avx2.c"
    , [ "16"; "--cil-r2z"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix16_z_r2z_bwd_avx2.c"
    , [ "16"; "--cil-r2z"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix32_z_r2z_avx2.c"
    , [ "32"; "--cil-r2z"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix32_z_r2z_bwd_avx2.c"
    , [ "32"; "--cil-r2z"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix64_z_r2z_avx2.c"
    , [ "64"; "--cil-r2z"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix64_z_r2z_bwd_avx2.c"
    , [ "64"; "--cil-r2z"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
    (* the real MONO rn1 (c2c_il.ml, 2026-09-30): the whole small real
       transform as one n1 body, fwd (r2c) + bwd (c2r), the n1 radices 2 to
       64 (2: the N = 2 real cell, the smallest the contract promises);
       zil/avx2/real/mono *)
  ; ( "radix2_z_rn1_avx2.c"
    , [ "2"; "--cil-rn1"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix2_z_rn1_bwd_avx2.c"
    , [ "2"; "--cil-rn1"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix3_z_rn1_avx2.c"
    , [ "3"; "--cil-rn1"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix3_z_rn1_bwd_avx2.c"
    , [ "3"; "--cil-rn1"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix4_z_rn1_avx2.c"
    , [ "4"; "--cil-rn1"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix4_z_rn1_bwd_avx2.c"
    , [ "4"; "--cil-rn1"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix5_z_rn1_avx2.c"
    , [ "5"; "--cil-rn1"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix5_z_rn1_bwd_avx2.c"
    , [ "5"; "--cil-rn1"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix6_z_rn1_avx2.c"
    , [ "6"; "--cil-rn1"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix6_z_rn1_bwd_avx2.c"
    , [ "6"; "--cil-rn1"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix7_z_rn1_avx2.c"
    , [ "7"; "--cil-rn1"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix7_z_rn1_bwd_avx2.c"
    , [ "7"; "--cil-rn1"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix8_z_rn1_avx2.c"
    , [ "8"; "--cil-rn1"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix8_z_rn1_bwd_avx2.c"
    , [ "8"; "--cil-rn1"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix9_z_rn1_avx2.c"
    , [ "9"; "--cil-rn1"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix9_z_rn1_bwd_avx2.c"
    , [ "9"; "--cil-rn1"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix10_z_rn1_avx2.c"
    , [ "10"; "--cil-rn1"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix10_z_rn1_bwd_avx2.c"
    , [ "10"; "--cil-rn1"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix11_z_rn1_avx2.c"
    , [ "11"; "--cil-rn1"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix11_z_rn1_bwd_avx2.c"
    , [ "11"; "--cil-rn1"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix12_z_rn1_avx2.c"
    , [ "12"; "--cil-rn1"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix12_z_rn1_bwd_avx2.c"
    , [ "12"; "--cil-rn1"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix13_z_rn1_avx2.c"
    , [ "13"; "--cil-rn1"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix13_z_rn1_bwd_avx2.c"
    , [ "13"; "--cil-rn1"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix14_z_rn1_avx2.c"
    , [ "14"; "--cil-rn1"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix14_z_rn1_bwd_avx2.c"
    , [ "14"; "--cil-rn1"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix15_z_rn1_avx2.c"
    , [ "15"; "--cil-rn1"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix15_z_rn1_bwd_avx2.c"
    , [ "15"; "--cil-rn1"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix16_z_rn1_avx2.c"
    , [ "16"; "--cil-rn1"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix16_z_rn1_bwd_avx2.c"
    , [ "16"; "--cil-rn1"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix17_z_rn1_avx2.c"
    , [ "17"; "--cil-rn1"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix17_z_rn1_bwd_avx2.c"
    , [ "17"; "--cil-rn1"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix19_z_rn1_avx2.c"
    , [ "19"; "--cil-rn1"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix19_z_rn1_bwd_avx2.c"
    , [ "19"; "--cil-rn1"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix21_z_rn1_avx2.c"
    , [ "21"; "--cil-rn1"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix21_z_rn1_bwd_avx2.c"
    , [ "21"; "--cil-rn1"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix22_z_rn1_avx2.c"
    , [ "22"; "--cil-rn1"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix22_z_rn1_bwd_avx2.c"
    , [ "22"; "--cil-rn1"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix23_z_rn1_avx2.c"
    , [ "23"; "--cil-rn1"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix23_z_rn1_bwd_avx2.c"
    , [ "23"; "--cil-rn1"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix25_z_rn1_avx2.c"
    , [ "25"; "--cil-rn1"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix25_z_rn1_bwd_avx2.c"
    , [ "25"; "--cil-rn1"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix26_z_rn1_avx2.c"
    , [ "26"; "--cil-rn1"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix26_z_rn1_bwd_avx2.c"
    , [ "26"; "--cil-rn1"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix27_z_rn1_avx2.c"
    , [ "27"; "--cil-rn1"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix27_z_rn1_bwd_avx2.c"
    , [ "27"; "--cil-rn1"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix29_z_rn1_avx2.c"
    , [ "29"; "--cil-rn1"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix29_z_rn1_bwd_avx2.c"
    , [ "29"; "--cil-rn1"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix31_z_rn1_avx2.c"
    , [ "31"; "--cil-rn1"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix31_z_rn1_bwd_avx2.c"
    , [ "31"; "--cil-rn1"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
    (* THE BLOCKED REAL MONO (2026-10-06): at 32 and 64 the mono is the c2c
       blocked construction on its own edges (c2c_il.ml emit_blocked with
       kind RN1): 4x8 at 32, the three-pass 4x4x4 at 64, both directions. The
       kernel race: 7-14% over the monolithic bodies at count 1 and 2, spills
       halved; the monolithic 32/64 bodies are deleted (the law: r32/r64
       never monolithic). The 8x8 twin at 64 tied at count 1, lost at count 2. *)
  ; ( "radix32_z_rn1b48_avx2.c"
    , [ "32"; "--cil-rn1"; "--cil-blocked"; "--cil-split"; "4.8"; "--cil-form-tag"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix32_z_rn1b48_bwd_avx2.c"
    , [ "32"; "--cil-rn1"; "--cil-blocked"; "--cil-split"; "4.8"; "--cil-form-tag"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix37_z_rn1_avx2.c"
    , [ "37"; "--cil-rn1"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix37_z_rn1_bwd_avx2.c"
    , [ "37"; "--cil-rn1"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix41_z_rn1_avx2.c"
    , [ "41"; "--cil-rn1"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix41_z_rn1_bwd_avx2.c"
    , [ "41"; "--cil-rn1"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix43_z_rn1_avx2.c"
    , [ "43"; "--cil-rn1"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix43_z_rn1_bwd_avx2.c"
    , [ "43"; "--cil-rn1"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix47_z_rn1_avx2.c"
    , [ "47"; "--cil-rn1"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix47_z_rn1_bwd_avx2.c"
    , [ "47"; "--cil-rn1"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
    (* 53, 59, 61 (2026-10-03): the odd primes between the chain's largest
       radix and the real Bluestein's floor -- the mono is their one engine *)
  ; ( "radix53_z_rn1_avx2.c"
    , [ "53"; "--cil-rn1"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix53_z_rn1_bwd_avx2.c"
    , [ "53"; "--cil-rn1"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix59_z_rn1_avx2.c"
    , [ "59"; "--cil-rn1"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix59_z_rn1_bwd_avx2.c"
    , [ "59"; "--cil-rn1"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix61_z_rn1_avx2.c"
    , [ "61"; "--cil-rn1"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix61_z_rn1_bwd_avx2.c"
    , [ "61"; "--cil-rn1"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix64_z_rn1b444_avx2.c"
    , [ "64"; "--cil-rn1"; "--cil-blocked"; "--cil-split3"; "4.4.4"; "--cil-form-tag"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix64_z_rn1b444_bwd_avx2.c"
    , [ "64"; "--cil-rn1"; "--cil-blocked"; "--cil-split3"; "4.4.4"; "--cil-form-tag"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
    (* the real FLAT leaf r1c (real_il.ml, 2026-09-30): real legs over contiguous
       columns -> the digit runs, fwd + bwd at the flat DIT's odd radices;
       zil/avx2/real/flat *)
  (* the real ROWS r2zr (2026-10-01): the real leaf over row-major rows, forward *)
  ; ( "radix4_z_r2zr_avx2.c"
    , [ "4"; "--cil-r2zr"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix6_z_r2zr_avx2.c"
    , [ "6"; "--cil-r2zr"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix8_z_r2zr_avx2.c"
    , [ "8"; "--cil-r2zr"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix10_z_r2zr_avx2.c"
    , [ "10"; "--cil-r2zr"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix12_z_r2zr_avx2.c"
    , [ "12"; "--cil-r2zr"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix16_z_r2zr_avx2.c"
    , [ "16"; "--cil-r2zr"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix32_z_r2zr_avx2.c"
    , [ "32"; "--cil-r2zr"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  (* its backward (2026-10-05): the CCE bins of every row -> its real samples, the 2D real c2r row pass *)
  ; ( "radix4_z_r2zr_bwd_avx2.c"
    , [ "4"; "--cil-r2zr"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix6_z_r2zr_bwd_avx2.c"
    , [ "6"; "--cil-r2zr"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix8_z_r2zr_bwd_avx2.c"
    , [ "8"; "--cil-r2zr"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix10_z_r2zr_bwd_avx2.c"
    , [ "10"; "--cil-r2zr"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix12_z_r2zr_bwd_avx2.c"
    , [ "12"; "--cil-r2zr"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix16_z_r2zr_bwd_avx2.c"
    , [ "16"; "--cil-r2zr"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix32_z_r2zr_bwd_avx2.c"
    , [ "32"; "--cil-r2zr"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix3_z_r1c_avx2.c"
    , [ "3"; "--cil-r1c"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix3_z_r1c_bwd_avx2.c"
    , [ "3"; "--cil-r1c"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix5_z_r1c_avx2.c"
    , [ "5"; "--cil-r1c"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix5_z_r1c_bwd_avx2.c"
    , [ "5"; "--cil-r1c"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix7_z_r1c_avx2.c"
    , [ "7"; "--cil-r1c"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix7_z_r1c_bwd_avx2.c"
    , [ "7"; "--cil-r1c"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix9_z_r1c_avx2.c"
    , [ "9"; "--cil-r1c"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix9_z_r1c_bwd_avx2.c"
    , [ "9"; "--cil-r1c"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix11_z_r1c_avx2.c"
    , [ "11"; "--cil-r1c"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix11_z_r1c_bwd_avx2.c"
    , [ "11"; "--cil-r1c"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix13_z_r1c_avx2.c"
    , [ "13"; "--cil-r1c"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix13_z_r1c_bwd_avx2.c"
    , [ "13"; "--cil-r1c"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix15_z_r1c_avx2.c"
    , [ "15"; "--cil-r1c"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix15_z_r1c_bwd_avx2.c"
    , [ "15"; "--cil-r1c"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix17_z_r1c_avx2.c"
    , [ "17"; "--cil-r1c"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix17_z_r1c_bwd_avx2.c"
    , [ "17"; "--cil-r1c"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix19_z_r1c_avx2.c"
    , [ "19"; "--cil-r1c"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix19_z_r1c_bwd_avx2.c"
    , [ "19"; "--cil-r1c"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix21_z_r1c_avx2.c"
    , [ "21"; "--cil-r1c"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix21_z_r1c_bwd_avx2.c"
    , [ "21"; "--cil-r1c"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix23_z_r1c_avx2.c"
    , [ "23"; "--cil-r1c"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix23_z_r1c_bwd_avx2.c"
    , [ "23"; "--cil-r1c"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix25_z_r1c_avx2.c"
    , [ "25"; "--cil-r1c"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix25_z_r1c_bwd_avx2.c"
    , [ "25"; "--cil-r1c"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix27_z_r1c_avx2.c"
    , [ "27"; "--cil-r1c"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix27_z_r1c_bwd_avx2.c"
    , [ "27"; "--cil-r1c"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix29_z_r1c_avx2.c"
    , [ "29"; "--cil-r1c"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix29_z_r1c_bwd_avx2.c"
    , [ "29"; "--cil-r1c"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix31_z_r1c_avx2.c"
    , [ "31"; "--cil-r1c"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix31_z_r1c_bwd_avx2.c"
    , [ "31"; "--cil-r1c"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix37_z_r1c_avx2.c"
    , [ "37"; "--cil-r1c"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix37_z_r1c_bwd_avx2.c"
    , [ "37"; "--cil-r1c"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix41_z_r1c_avx2.c"
    , [ "41"; "--cil-r1c"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix41_z_r1c_bwd_avx2.c"
    , [ "41"; "--cil-r1c"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix43_z_r1c_avx2.c"
    , [ "43"; "--cil-r1c"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix43_z_r1c_bwd_avx2.c"
    , [ "43"; "--cil-r1c"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix47_z_r1c_avx2.c"
    , [ "47"; "--cil-r1c"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix47_z_r1c_bwd_avx2.c"
    , [ "47"; "--cil-r1c"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
    (* the real flat DIT's Hermitian last stage (2026-10-01): t2csgh stores the
       half spectrum, t2csght reads it; the flat tail radices; zil/avx2/real/flat/herm *)
  ; ( "radix3_z_t2csgh_avx2.c"
    , [ "3"; "--cil-t2csgh"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix3_z_t2csght_bwd_avx2.c"
    , [ "3"; "--cil-t2csght"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix5_z_t2csgh_avx2.c"
    , [ "5"; "--cil-t2csgh"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix5_z_t2csght_bwd_avx2.c"
    , [ "5"; "--cil-t2csght"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix7_z_t2csgh_avx2.c"
    , [ "7"; "--cil-t2csgh"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix7_z_t2csght_bwd_avx2.c"
    , [ "7"; "--cil-t2csght"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix9_z_t2csgh_avx2.c"
    , [ "9"; "--cil-t2csgh"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix9_z_t2csght_bwd_avx2.c"
    , [ "9"; "--cil-t2csght"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix11_z_t2csgh_avx2.c"
    , [ "11"; "--cil-t2csgh"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix11_z_t2csght_bwd_avx2.c"
    , [ "11"; "--cil-t2csght"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix13_z_t2csgh_avx2.c"
    , [ "13"; "--cil-t2csgh"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix13_z_t2csght_bwd_avx2.c"
    , [ "13"; "--cil-t2csght"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix15_z_t2csgh_avx2.c"
    , [ "15"; "--cil-t2csgh"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix15_z_t2csght_bwd_avx2.c"
    , [ "15"; "--cil-t2csght"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix17_z_t2csgh_avx2.c"
    , [ "17"; "--cil-t2csgh"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix17_z_t2csght_bwd_avx2.c"
    , [ "17"; "--cil-t2csght"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix19_z_t2csgh_avx2.c"
    , [ "19"; "--cil-t2csgh"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix19_z_t2csght_bwd_avx2.c"
    , [ "19"; "--cil-t2csght"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix21_z_t2csgh_avx2.c"
    , [ "21"; "--cil-t2csgh"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix21_z_t2csght_bwd_avx2.c"
    , [ "21"; "--cil-t2csght"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix23_z_t2csgh_avx2.c"
    , [ "23"; "--cil-t2csgh"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix23_z_t2csght_bwd_avx2.c"
    , [ "23"; "--cil-t2csght"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix25_z_t2csgh_avx2.c"
    , [ "25"; "--cil-t2csgh"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix25_z_t2csght_bwd_avx2.c"
    , [ "25"; "--cil-t2csght"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix27_z_t2csgh_avx2.c"
    , [ "27"; "--cil-t2csgh"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix27_z_t2csght_bwd_avx2.c"
    , [ "27"; "--cil-t2csght"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix29_z_t2csgh_avx2.c"
    , [ "29"; "--cil-t2csgh"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix29_z_t2csght_bwd_avx2.c"
    , [ "29"; "--cil-t2csght"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix31_z_t2csgh_avx2.c"
    , [ "31"; "--cil-t2csgh"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix31_z_t2csght_bwd_avx2.c"
    , [ "31"; "--cil-t2csght"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix37_z_t2csgh_avx2.c"
    , [ "37"; "--cil-t2csgh"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix37_z_t2csght_bwd_avx2.c"
    , [ "37"; "--cil-t2csght"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix41_z_t2csgh_avx2.c"
    , [ "41"; "--cil-t2csgh"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix41_z_t2csght_bwd_avx2.c"
    , [ "41"; "--cil-t2csght"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix43_z_t2csgh_avx2.c"
    , [ "43"; "--cil-t2csgh"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix43_z_t2csght_bwd_avx2.c"
    , [ "43"; "--cil-t2csght"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix47_z_t2csgh_avx2.c"
    , [ "47"; "--cil-t2csgh"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
  ; ( "radix47_z_t2csght_bwd_avx2.c"
    , [ "47"; "--cil-t2csght"; "--cil-bwd"; "--isa"; "avx2"; "--uarch"; "raptor_lake_avx2" ] )
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
   (need an emitter suffix knob). (radix256_r2c_term_ls_r8_avx512, once
   excluded for its --isa-after---emit-c argv, was referenced by nothing and
   was deleted 2026-11-04 with the other dead split codelets; the four
   `_spec1024/4096` edge cells left with the `_spec` twins the same day.) *)
let oop_edges_avx2_cells : (string * string list) list =
    [ "vfft_k1_mono64_il_bwd_avx2.c", [ "64"; "--k1-mono"; "--k1-il"; "--k1-sw"; "--isa"; "avx2" ]
    ; "vfft_k1_mono64_il_fwd_avx2.c", [ "64"; "--k1-mono"; "--k1-il"; "--isa"; "avx2" ]
    ]
;;

let oop_edges_avx512_cells : (string * string list) list =
    [ "vfft_k1_mono64_il_bwd_avx512.c", [ "64"; "--k1-mono"; "--k1-il"; "--k1-sw"; "--isa"; "avx512" ]
    ; "vfft_k1_mono64_il_fwd_avx512.c", [ "64"; "--k1-mono"; "--k1-il"; "--isa"; "avx512" ]
    ]
;;

(* ── zil at a SECOND ISA (2026-09-26): DERIVED from the avx2 cells ──
   The literal avx2 lists above stay the one source of truth; an ISA is a
   PARAMETER: retarget the typed cell (isa/uarch fields of Codelet.t, then
   to_argv — so the round-trip law holds by construction) and rename the file
   suffix.  A cell the generator cannot emit at that width is DECLARED absent
   (zil_isa_gap restates each emitter refusal with its reason, which the
   registry prints), never silently dropped: gen_set stops on any refusal
   not declared there. *)
let zil_isa_uarch = [ "avx2", "raptor_lake_avx2"; "avx512", "sapphire_rapids_avx512" ]
let zil_isas = List.map fst zil_isa_uarch

let isa_vw (isa : string) : int =
  match isa with
  | "avx2" -> 4
  | "avx512" -> 8
  | i -> failwith ("Corpus: unknown zil isa " ^ i)
;;

let zil_isa_gap ~(isa : string) (c : Codelet.t) : string option =
  (* every zil cell the emitters refuse at this width, each with the refusal
     it restates. ONE is a law: the ZTURN-T lane lattices t0tp/tld write or
     read R-runs as whole blocks, so R % VW == 0 (cascade_z.ml refuses it).
     The others are absent PENDING their wider edges, not by law; each
     emitter renders the edge at 256 bits (and 128 for its narrow arms) only:
       - the real pair r2z / t2h / t2m and the real rows r2zr (real_il.ml);
       - the n1c column leaf's 128-bit half stores n1ch and n1cb48h / n1cb84h
         / n1cb88h (c2c_il.ml, --cil-st128): whether the 512-bit leaf wants
         its plane stores split waits on a Zen 4 measurement
         (gauntlet/half_store_bench.sh).
     Every other zil kind is emitted at both widths. *)
  let vw = isa_vw isa in
  let pending why = Some (Printf.sprintf "no %d-bit edge yet: %s" (64 * vw) why) in
  match c.Codelet.kind with
  | Codelet.Zsplit { k = Codelet.(T0tp | T0tpb | Tld | Tldb) } when c.Codelet.radix mod vw <> 0 ->
    Some (Printf.sprintf "codelet_zsplit: t0tp/tld radix %d is not a whole number of %d-lane blocks (law: R %% VW == 0)" c.Codelet.radix vw)
  | Codelet.Cil { form = Codelet.(Cil_r2z | Cil_t2h | Cil_t2m | Cil_r2zr); _ } when vw <> 4 ->
    pending "real_il emits the real pair and real rows kinds for the 256-bit ISA only"
  | Codelet.Cil { st128 = true; _ } when vw <> 4 ->
    pending "codelet_cil renders the half stores (--cil-st128) at 256 bits, the halves of a column pair; split stores at 512 bits wait on a Zen 4 measurement (gauntlet/half_store_bench.sh)"
  (* the ROW family (2026-11-08): rowswap was raced at 256 bits only -- at 512 bits n1ccs serves until it is
     raced there; the row form needs the radix a multiple of per^2 and its stage-1 split to factor p = R/per
     with m1 a multiple of per (c2c_il.ml emit_row), so the 256-bit cells 4, 12 and the splits 4.4 / 4.8 have
     no 512-bit twin (zil_pure_cells_avx512_only carries that ISA's own row cells) *)
  | Codelet.Cil { form = Codelet.Cil_rowswap; _ } when vw <> 4 ->
    Some "the row kernels' whole-load form (rowswap) was raced at 256 bits only (2026-11-08); at 512 bits the lane-load n1ccs construction serves until it is raced there"
  | Codelet.Cil { form = Codelet.Cil_row; split; _ } ->
    let per = vw / 2 in
    let r = c.Codelet.radix in
    if r mod (per * per) <> 0
    then Some (Printf.sprintf "codelet_cil: the row kind needs radix %d to be a multiple of %d (per^2) at %d bits" r (per * per) (64 * vw))
    else (
      match split with
      | Some (m1, p1) when m1 * p1 <> r / per || m1 mod per <> 0 ->
        Some (Printf.sprintf "codelet_cil: --cil-split %d.%d does not factor the row kind's stage 1 at %d bits (p = %d, m1 a multiple of %d)" m1 p1 (64 * vw) (r / per) per)
      | _ -> None)
  | _ -> None
;;

(* the 512-bit row cells of their own (2026-11-08): the row form at 32 and 64 (the stage-1 splits that won at
   256 bits do not factor p = 8 / 16, and at 512 bits stage 1 holds 8 / 16 registers: no split needed) and the
   split form at 128 (owner: AVX-512 has a 128 row kernel; 4.8 counted 5.4 instructions per point). Not raced:
   this machine has no AVX-512. *)
let zil_pure_cells_avx512_only : (string * string list) list =
  [ ( "radix32_z_row_avx512.c", [ "32"; "--cil-row"; "--isa"; "avx512"; "--uarch"; "sapphire_rapids_avx512" ] )
  ; ( "radix32_z_row_bwd_avx512.c", [ "32"; "--cil-row"; "--cil-bwd"; "--isa"; "avx512"; "--uarch"; "sapphire_rapids_avx512" ] )
  ; ( "radix64_z_row_avx512.c", [ "64"; "--cil-row"; "--isa"; "avx512"; "--uarch"; "sapphire_rapids_avx512" ] )
  ; ( "radix64_z_row_bwd_avx512.c", [ "64"; "--cil-row"; "--cil-bwd"; "--isa"; "avx512"; "--uarch"; "sapphire_rapids_avx512" ] )
  ; ( "radix128_z_rowb48_avx512.c", [ "128"; "--cil-row"; "--cil-split"; "4.8"; "--isa"; "avx512"; "--uarch"; "sapphire_rapids_avx512" ] )
  ; ( "radix128_z_rowb48_bwd_avx512.c", [ "128"; "--cil-row"; "--cil-split"; "4.8"; "--cil-bwd"; "--isa"; "avx512"; "--uarch"; "sapphire_rapids_avx512" ] )
  ]
;;

let chop_isa_suffix (name : string) : string =
  let sfx = "_avx2.c" in
  let ln = String.length name and ls = String.length sfx in
  if ln >= ls && String.sub name (ln - ls) ls = sfx
  then String.sub name 0 (ln - ls)
  else failwith ("Corpus: zil cell not named *_avx2.c: " ^ name)
;;

(* (cells emitted, cells declared absent with the reason) *)
let zil_retarget ~(isa : string) (cells : (string * string list) list)
  : (string * string list) list * (string * string) list
  =
  let uarch = List.assoc isa zil_isa_uarch in
  List.fold_right
    (fun (file, tail) (ok, gap) ->
       let c = Codelet.of_argv (tail @ [ "--emit-c" ]) in
       let c = { c with Codelet.isa = Some isa; uarch = Some uarch } in
       let file' = chop_isa_suffix file ^ "_" ^ isa ^ ".c" in
       match zil_isa_gap ~isa c with
       | Some why -> ok, (file', why) :: gap
       | None ->
         let argv = Codelet.to_argv c in
         let tail' = List.filter (fun a -> a <> "--emit-c") argv in
         (file', tail') :: ok, gap)
    cells
    ([], [])
;;

let zil_quadrant_isa (q : string) : (string * string) option =
  (* "zil-pure-avx512" -> Some ("zil-pure", "avx512"); the bare names are avx2 *)
  List.find_map
    (fun isa ->
       let sfx = "-" ^ isa in
       let lq = String.length q and ls = String.length sfx in
       if lq > ls && String.sub q (lq - ls) ls = sfx
       && (String.sub q 0 (lq - ls) = "zil-pure" || String.sub q 0 (lq - ls) = "zil-boundary")
       then Some (String.sub q 0 (lq - ls), isa)
       else None)
    (List.filter (fun i -> i <> "avx2") zil_isas)
;;

let zil_base_cells (base : string) =
  if base = "zil-pure" then zil_pure_cells else zil_boundary_cells
;;

let zil_absent (q : string) : (string * string) list =
  match zil_quadrant_isa q with
  | Some (base, isa) -> snd (zil_retarget ~isa (zil_base_cells base))
  | None -> []
;;

let matrix_files (quadrant : string) : (string * string list) list =
  match quadrant with
  | q when zil_quadrant_isa q <> None ->
    (match zil_quadrant_isa q with
     | Some (base, isa) ->
       fst (zil_retarget ~isa (zil_base_cells base))
       @ (if base = "zil-pure" && isa = "avx512" then zil_pure_cells_avx512_only else [])
     | None -> assert false)
  | "oop-edges-avx2" -> oop_edges_avx2_cells
  | "oop-edges-avx512" -> oop_edges_avx512_cells
  | "ztt-avx2" -> zts_cells "avx2"
  | "zil-boundary" -> zil_boundary_cells
  | "zil-pure" -> zil_pure_cells
  | q -> failwith ("Corpus.files: unknown quadrant " ^ q)
;;

let quadrants =
  [ "oop-edges-avx2"
  ; "oop-edges-avx512"
  ; "ztt-avx2"
  ; "zil-boundary"
  ; "zil-pure"
  ; "zil-boundary-avx512"
  ; "zil-pure-avx512"
  ]
;;

(* directory under codelets/ for each quadrant. The SPLIT library's families
   live under codelets/split/<family>/<isa> (2026-11-04: one folder per
   library, beside zil/ for the IL library); the zil arms are reached only by
   the uniqueness law, zil files land through zil_folder. *)
let dir_of_quadrant (q : string) : string =
  match q with
  | "oop-edges-avx2" -> "split/oop/avx2"
  | "oop-edges-avx512" -> "split/oop/avx512"
  | "zil-boundary" -> "zil/avx2/boundary_split"
  | "zil-pure" -> "zil/avx2/pure_il"
  | "zil-boundary-avx512" -> "zil/avx512/boundary_split"
  | "zil-pure-avx512" -> "zil/avx512/pure_il"
  | _ ->
    (match String.split_on_char '-' q with
     | [ fam; isa ] -> "split/" ^ fam ^ "/" ^ isa
     | _ -> failwith ("Corpus.dir_of_quadrant: " ^ q))
;;

(* expected_counts: DELETED at M10 — zero consumers repo-wide. *)

(* ── M10: the typed layer — the corpus AS Codelet.t values ── *)

type cell =
  { file : string
  ; c : Codelet.t
  }


(* THE INTERLEAVED TREE'S FOLDERS (2026-09-24, docs/roadmap/il_codelet_reorg.md):
   a zil file's folder follows its KIND, not its quadrant -- shared/ (n1),
   shared/col[/blocked] (the 2D column stage), mono/, pair2p[/blocked|/tangent],
   rows/ (the 2D row kinds), chain3/, flat[/odd_mid], ztt/. gen_set places every
   file through dir_of_file; the registry emitters read the tables, not the
   disk. None = not a zil file: the quadrant's own folder applies. *)
let zil_folder (name : string) : string option =
  let starts_with s pre =
    let ls = String.length s and lp = String.length pre in
    ls >= lp && String.sub s 0 lp = pre
  in
  let ends_with s suf =
    let ls = String.length s and lf = String.length suf in
    ls >= lf && String.sub s (ls - lf) lf = suf
  in
  match List.find_opt (fun isa -> ends_with name ("_" ^ isa ^ ".c")) zil_isas with
  | None -> None
  | Some isa ->
  let sfxlen = String.length isa + 3 in
  let z f = Some ("zil/" ^ isa ^ "/" ^ f) in
  if starts_with name "vfft_k1_mono64_il_" then z "mono"
  else if not (starts_with name "radix") then None
  else
    match String.index_opt name '_' with
    | Some i when i + 3 < String.length name - sfxlen && String.sub name i 3 = "_z_" ->
      let kind = String.sub name (i + 3) (String.length name - i - 3 - sfxlen) in
      let base =
        if ends_with kind "_bwd" then String.sub kind 0 (String.length kind - 4) else kind
      in
      let is_digits t = t <> "" && String.to_seq t |> Seq.for_all (fun c -> c >= '0' && c <= '9') in
      let blocked pre =
        starts_with base pre
        && (let t = String.sub base (String.length pre) (String.length base - String.length pre) in
            t = "" || is_digits t)
      in
      z
        (if base = "n1" then "shared"
         else if base = "n1c" || base = "t2c" then "shared/col"
         else if base = "n1ch"
                 || (starts_with base "n1cb" && ends_with base "h"
                     && is_digits (String.sub base 4 (String.length base - 5)))
         then "shared/col/half"
         else if blocked "n1cb" || blocked "t2cb" then "shared/col/blocked"
         (* the 2D tier's batched ROW kernels (2026-11-08): one folder per construction under rows/ --
            n1ccs (lane loads), rowswap (whole-vector loads and swaps), row (one row per iteration, with its
            split forms rowb<m1><p1>) *)
         else if base = "n1ccs" then "rows/n1ccs"
         else if starts_with base "rowswap" then "rows/rowswap"   (* rowswap, rowswapt (turned stores) *)
         else if starts_with base "row" then "rows/row"           (* row, rowb<m1><p1>, and their turned twins ...t *)
         else if List.mem base [ "n1r"; "n1tr"; "t2r"; "t2tr"; "n1rtan"; "n1trtan"; "t2rtan"; "t2trtan" ]
         then "rows"
         else if List.mem base [ "n1t"; "t2"; "t2t"; "n1_ct"; "n1t_ct"; "t2_ct"; "t2t_ct" ] then "pair2p"
         else if blocked "n1tb" || blocked "n1b" || blocked "t2bt" || blocked "t2b" then "pair2p/blocked"
         else if base = "t2tg" then "chain3"
         else if List.mem base [ "t2cp"; "t2cs"; "t2csg"; "t2csgn"; "t2csgt"; "t2csgnt" ] then "flat"
         else if base = "msz" || base = "mszt" then "flat/odd_mid"
         else if List.mem base [ "t0tp"; "tmg"; "tlf"; "tlfi"; "t0d"; "tmgd"; "tld" ] then "ztt"
         else if base = "rn1" || blocked "rn1b" then "real/mono"
         else if base = "r1c" then "real/flat"
         else if base = "r2zr" then "real/rows"
         else if base = "t2csgh" || base = "t2csght" then "real/flat/herm"
         else if base = "t2h" || base = "t2m" || base = "r2z" then "real/pair"
         else "pair2p/tangent")
    | _ -> None

let dir_of_file (q : string) (name : string) : string =
  match zil_folder name with Some d -> d | None -> dir_of_quadrant q

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
