(* coverage.ml — THE single source of truth for codelet coverage.
 *
 * Section 39: radix coverage used to be defined three times (bash in
 * regen_codelets.sh / generate_codelets.sh, OCaml in emit_registry_h),
 * with dune unable to watch the bash side. This module is now the only
 * definition; consumers:
 *   - bin/gen_set.ml      (the in-process tree generator / driver)
 *   - bin/emit_registry_h (registry coverage)
 *   - scripts             (thin invokers of gen_set; carry no lists)
 *
 * Each quadrant maps to (filename, argv_tail) pairs where argv_tail is
 * the exact gen_radix CLI the codelet is generated with (sans exe name
 * and sans --emit-c, which the driver appends). Structure rules that
 * are CODE, not coverage — the 18-family matrix, the spec stride
 * formula RV = R*8, strided's single-stage restriction — live here as
 * functions, not as data to keep in sync. *)

let ip_radices =
  [ 2; 3; 4; 5; 6; 7; 8; 10; 11; 12; 13; 16; 17; 19; 20; 25; 32; 64 ]

let oop_n1_radices =
  [
    2;
    3;
    4;
    5;
    6;
    7;
    8;
    9;
    10;
    11;
    12;
    13;
    14;
    15;
    16;
    17;
    19;
    20;
    25;
    32;
    64;
    128;
  ]

let t1p_radices = [ 4; 7; 8; 13; 16; 32; 64 ]
let spec_radices = [ 7; 13; 32 ] (* stride formula: rv = r * 8 *)
let strided_radices_avx2 = [ 4; 8; 12; 16; 20; 32; 64 ]
let strided_radices_avx512 = [ 8; 16; 32; 64 ]

(* The in-place 18-family matrix: n1 fwd/bwd plus
 * {t1,t1s} x {dit,dif} x {fwd,bwd} x {flat,log3}. *)
let ip_families (isa : string) : (string * string list) list =
  let n1 =
    [
      ("n1_fwd", [ "--in-place"; "--isa"; isa; "--su" ]);
      ("n1_bwd", [ "--in-place"; "--isa"; isa; "--su"; "--bwd" ]);
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
                      @ targ @ darg @ barg @ larg
                    in
                    (fam, args))
                  [ ("", []); ("_log3", [ "--log3" ]) ])
              [ ("_fwd", []); ("_bwd", [ "--bwd" ]) ])
          [ ("_dit", []); ("_dif", [ "--dif" ]) ])
      [ ("t1", []); ("t1s", [ "--t1s" ]) ]
  in
  n1 @ t1

let oop_base (isa : string) : string list =
  [
    "--oop";
    "--oop-buffer-oop";
    "--oop-load";
    "UG";
    "--oop-store";
    "UG";
    "--isa";
    isa;
  ]

(* (filename, argv_tail) pairs per quadrant. argv_tail excludes the exe
 * name and --emit-c. Filenames match the committed tree exactly. *)
let files (quadrant : string) : (string * string list) list =
  match quadrant with
  | "inplace-avx2" | "inplace-avx512" ->
      let isa = if quadrant = "inplace-avx2" then "avx2" else "avx512" in
      List.concat_map
        (fun r ->
          List.map
            (fun (fam, args) ->
              (Printf.sprintf "r%d_%s.c" r fam, string_of_int r :: args))
            (ip_families isa))
        ip_radices
  | "oop-avx2" | "oop-avx512" ->
      let isa = if quadrant = "oop-avx2" then "avx2" else "avx512" in
      let base = oop_base isa in
      let n1 =
        List.map
          (fun r ->
            (Printf.sprintf "radix%d_n1_oop_%s.c" r isa, string_of_int r :: base))
          oop_n1_radices
      in
      let t1p =
        List.concat_map
          (fun r ->
            [
              ( Printf.sprintf "radix%d_t1p_oop_%s.c" r isa,
                (string_of_int r :: base) @ [ "--twiddled-pos" ] );
              ( Printf.sprintf "radix%d_t1p_log3_oop_%s.c" r isa,
                (string_of_int r :: base) @ [ "--twiddled-pos"; "--log3" ] );
            ])
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
            [
              ( Printf.sprintf "radix%d_n1_oop_%s_spec.c" r isa,
                (string_of_int r :: base) @ opt
                @ [ "--oop-strides"; strides_n1 ] );
              ( Printf.sprintf "radix%d_t1p_oop_%s_spec.c" r isa,
                (string_of_int r :: base) @ opt
                @ [ "--oop-strides"; strides_t1p; "--twiddled-pos" ] );
              ( Printf.sprintf "radix%d_t1p_log3_oop_%s_spec.c" r isa,
                (string_of_int r :: base) @ opt
                @ [ "--oop-strides"; strides_t1p; "--twiddled-pos"; "--log3" ]
              );
            ])
          spec_radices
      in
      n1 @ t1p @ extras
  | "strided-avx2" | "strided-avx512" ->
      let isa = if quadrant = "strided-avx2" then "avx2" else "avx512" in
      let radices =
        if isa = "avx2" then strided_radices_avx2 else strided_radices_avx512
      in
      List.concat_map
        (fun r ->
          [
            ( Printf.sprintf "r%d_n1_fwd_strided.c" r,
              [ string_of_int r; "--strided"; "--isa"; isa ] );
            ( Printf.sprintf "r%d_n1_bwd_strided.c" r,
              [ string_of_int r; "--strided"; "--isa"; isa; "--bwd" ] );
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
          ( Printf.sprintf "radix%d_r2cf_%s.c" r isa,
            [ string_of_int r; "--r2cf"; "--isa"; isa; "--su" ] ))
        leaf_only
      @ List.concat_map
          (fun r ->
            [
              ( Printf.sprintf "radix%d_r2cf_%s.c" r isa,
                [ string_of_int r; "--r2cf"; "--isa"; isa; "--su" ] );
              ( Printf.sprintf "radix%d_hc2hc_dit_fwd_%s.c" r isa,
                [ string_of_int r; "--hc2hc"; "--t1s"; "--isa"; isa; "--su" ] );
              ( Printf.sprintf "radix%d_hc2hc_dif_fwd_%s.c" r isa,
                [
                  string_of_int r;
                  "--hc2hc";
                  "--dif";
                  "--t1s";
                  "--isa";
                  isa;
                  "--su";
                ] );
              ( Printf.sprintf "radix%d_hc2c_dit_fwd_%s.c" r isa,
                [ string_of_int r; "--hc2c"; "--t1s"; "--isa"; isa; "--su" ] );
              (* log3 variants (section 62: hc2cf2 = hc2c + log3). FFTW's hc2cf2
               * family is literally hc2c generated with -twiddle-log3; here the
               * --log3 flag composes with --t1s the same way (verified: 7->3
               * twiddle slots, op counts match FFTW hc2cf2_8 at 74 add/30 fma).
               * The log3 twiddle stage (hc2hc) and the log3 NATURAL terminator
               * (hc2c-nat, the 6-ptr mirror-pair ABI the rfft executor's stage-0
               * actually calls — NOT the packed 4-ptr hc2c). *)
              ( Printf.sprintf "radix%d_hc2hc_dit_log3_fwd_%s.c" r isa,
                [
                  string_of_int r;
                  "--hc2hc";
                  "--log3";
                  "--t1s";
                  "--isa";
                  isa;
                  "--su";
                ] );
              ( Printf.sprintf "radix%d_hc2c_nat_log3_fwd_%s.c" r isa,
                [
                  string_of_int r;
                  "--hc2c-nat";
                  "--log3";
                  "--t1s";
                  "--isa";
                  isa;
                  "--su";
                ] );
              ( Printf.sprintf "radix%d_hc2c_nat_fwd_%s.c" r isa,
                [ string_of_int r; "--hc2c-nat"; "--t1s"; "--isa"; isa; "--su" ]
              );
              ( Printf.sprintf "radix%d_hc2hc_dit_rng_fwd_%s.c" r isa,
                [
                  string_of_int r;
                  "--hc2hc";
                  "--ranged";
                  "--t1s";
                  "--isa";
                  isa;
                  "--su";
                ] );
              ( Printf.sprintf "radix%d_hc2c_nat_rng_fwd_%s.c" r isa,
                [
                  string_of_int r;
                  "--hc2c-nat";
                  "--ranged";
                  "--t1s";
                  "--isa";
                  isa;
                  "--su";
                ] );
            ])
          radices
  | "trig-avx2" | "trig-avx512" ->
      (* Real-to-real trig family (notebook section 51): lean 3-arg ABI
       * (in, out, K), constant hoisting on, consumed by core/dct.h-style
       * plan shells. rdft deferred: complex output, generic ABI. *)
      let isa = if quadrant = "trig-avx2" then "avx2" else "avx512" in
      let kind_sizes =
        [
          ("dct2", [ 8; 16; 32; 64 ]);
          ("dct3", [ 8; 16; 32; 64 ]);
          ("dct4", [ 8; 16; 32; 64 ]);
          ("dst2", [ 8; 16; 32; 64 ]);
          ("dst3", [ 8; 16; 32; 64 ]);
          ("dst4", [ 8; 16; 32; 64 ]);
          ("dht", [ 8; 16; 32; 64 ]);
          (* Boundary kinds run at their logical-extension sizes:
         dct1 at N needs M = 2(N-1) radix-coverable -> N = 2^k+1
         (Chebyshev grids); dst1 needs M = 2(N+1) -> N = 2^k-1. *)
          ("dct1", [ 5; 9; 17; 33 ]);
          ("dst1", [ 3; 7; 15; 31 ]);
        ]
      in
      List.concat_map
        (fun (kind, sizes) ->
          List.map
            (fun n ->
              ( Printf.sprintf "radix%d_%s_%s.c" n kind isa,
                [ string_of_int n; "--" ^ kind; "--isa"; isa; "--su" ] ))
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
          ( Printf.sprintf "radix%d_r2cb_%s.c" r isa,
            [ string_of_int r; "--r2cb"; "--isa"; isa; "--su" ] ))
        leaf_only
      @ List.concat_map
          (fun r ->
            [
              ( Printf.sprintf "radix%d_r2cb_%s.c" r isa,
                [ string_of_int r; "--r2cb"; "--isa"; isa; "--su" ] );
              ( Printf.sprintf "radix%d_hc2hc_dif_bwd_%s.c" r isa,
                [
                  string_of_int r;
                  "--hc2hc";
                  "--dif";
                  "--bwd";
                  "--t1s";
                  "--isa";
                  isa;
                  "--su";
                ] );
              ( Printf.sprintf "radix%d_hc2hc_dif_log3_bwd_%s.c" r isa,
                [
                  string_of_int r;
                  "--hc2hc";
                  "--dif";
                  "--bwd";
                  "--log3";
                  "--t1s";
                  "--isa";
                  isa;
                  "--su";
                ] );
              ( Printf.sprintf "radix%d_hc2hc_dif_rng_bwd_%s.c" r isa,
                [
                  string_of_int r;
                  "--hc2hc";
                  "--dif";
                  "--bwd";
                  "--ranged";
                  "--t1s";
                  "--isa";
                  isa;
                  "--su";
                ] );
              (* c2r NATURAL INITIATOR (stage-0 split-input, inverse of the rfft
               * forward hc2c_nat terminator): reads the SPLIT half-spectrum and
               * feeds the packed c2r cascade with no repack. --bwd --dif matches
               * the cascade orientation; verified to invert the forward at 7e-15. *)
              ( Printf.sprintf "radix%d_hc2c_nat_log3_bwd_%s.c" r isa,
                [
                  string_of_int r;
                  "--hc2c-nat";
                  "--bwd";
                  "--dif";
                  "--log3";
                  "--t1s";
                  "--isa";
                  isa;
                  "--su";
                ] );
              ( Printf.sprintf "radix%d_hc2c_nat_bwd_%s.c" r isa,
                [
                  string_of_int r;
                  "--hc2c-nat";
                  "--bwd";
                  "--dif";
                  "--t1s";
                  "--isa";
                  isa;
                  "--su";
                ] );
              ( Printf.sprintf "radix%d_hc2c_nat_rng_bwd_%s.c" r isa,
                [
                  string_of_int r;
                  "--hc2c-nat";
                  "--bwd";
                  "--dif";
                  "--ranged";
                  "--t1s";
                  "--isa";
                  isa;
                  "--su";
                ] );
            ])
          radices
  | q -> failwith ("Coverage.files: unknown quadrant " ^ q)

let quadrants =
  [
    "inplace-avx2";
    "inplace-avx512";
    "oop-avx2";
    "oop-avx512";
    "strided-avx2";
    "strided-avx512";
    "trig-avx2";
    "trig-avx512";
    "rfft-avx2";
    "rfft-avx512";
    "c2r-avx2";
    "c2r-avx512";
  ]

(* directory under codelets/ for each quadrant *)
let dir_of_quadrant (q : string) : string =
  match String.split_on_char '-' q with
  | [ fam; isa ] -> fam ^ "/" ^ isa
  | _ -> failwith ("Coverage.dir_of_quadrant: " ^ q)

let expected_counts : (string * int) list =
  List.map (fun q -> (q, List.length (files q))) quadrants
