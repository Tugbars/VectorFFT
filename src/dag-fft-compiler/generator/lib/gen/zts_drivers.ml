(* zts_drivers.ml — zturn-ts (2026-11-04): ZTURN-T's run-contiguous DIT
 * arrangement on the SPLIT library's own planes, and its FUSED DRIVERS.
 *
 *   S (fam "zts", cascade_z.ml P_split): K = 1 on TRUE split planes re[] /
 *     im[]. The lanes are 4 consecutive columns of ONE transform. The ingest
 *     loads legs at stride N/R0 (no deinterleave) and turns 4 columns x R0
 *     digits per plane (TR4) into run-contiguous BLOCK-plane stores through
 *     rb[]; the mids are ZTURN-T's block combine; the last stage stores
 *     natural order straight into re / im — no shuffle where IL pays REINT.
 *     The interior is the 64-B [re x4][im x4] block plane (the scratch): two
 *     true split planes as interior cost 1.3x on the ingest and 1.7x on an
 *     R = 8 last stage (a column's lines 32 KB apart, two RFO streams).
 *   V (fam "ztv", P_vec): lane-major K data[e*K + t] (lane = transform), K
 *     any multiple of VW (2026-11-04: K = 8 named by the owner). The pipeline
 *     runs per LANE GROUP of VW lanes: the ingest loads the group's vector of
 *     every element at the runtime element pitch ep = K (its lane offset
 *     folded into the base pointer), the interior planes hold ONE group
 *     (N*VW doubles each), the terminator stores the group back at pitch K.
 *     Every column is ONE vector per plane: no turn anywhere, twiddles
 *     broadcast from (c, s) scalar pairs.
 *
 * Driver ABI: S (in_re, in_im, out_re, out_im, plane, tw, rb, tile) with a
 * `_batch` twin (..., size_t K) looping K CONTIGUOUS transforms at offset
 * t*N transform-major (one scratch plane); V (in_re, in_im, out_re, out_im,
 * plane_re, plane_im, tw, rb, tile, K) with K the lane count (a multiple of
 * VW; the group loop is the driver's). Both run the pipeline IN THE SCRATCH
 * (ZTURN-T's `plane` mode) and the terminator tlfi writes the caller's
 * planes: in place when in == out. There is no `dest` placement for split
 * (two separate caller planes cannot host the block interior, and a
 * terminator writing the split output over the blocks it reads would
 * clobber unconsumed data).
 * tw = ONE contiguous stream in stage order: S = ZTURN-T's quad records
 * [c x4][s x4] per (column quad, leg); V = (c, s) pairs per (column, leg);
 * both 2*(R-1)*L doubles per stage (cascade_z.ml %TWF). rb[] = ZTURN-T's
 * run-base table (rb[digitrev(j, chain+1)] = j). tile = ZTURN-T's law
 * (vfft_ztt_tile_legal). The geometry is Ztt_drivers.geom, unchanged; the
 * cell list is ZTURN-T's (Ztt_drivers.cells, natural_ok). *)

type fam =
  | S
  | V

let fam_tag = function
  | S -> "zts"
  | V -> "ztv"
;;

(* doubles per complex per plane *)
let scale ~(isa : Isa.t) = function
  | S -> 1
  | V -> isa.Isa.vec_width
;;

(* the kind string cascade_z.ml parses: "ts"|"tv" ^ base minus its 't' ^ "b" *)
let kind_name ~(fam : fam) (base : string) ~(bwd : bool) : string =
  (match fam with
   | S -> "ts"
   | V -> "tv")
  ^ String.sub base 1 (String.length base - 1)
  ^ if bwd then "b" else ""
;;

let driver_name ~(isa : string) ~(fam : fam) (n : int) (ch : int list) ~(bwd : bool) : string =
  Printf.sprintf
    "%s_%s_%s_plane_%s"
    (fam_tag fam)
    (Ztt_drivers.tag n ch)
    (if bwd then "bwd" else "fwd")
    isa
;;

let driver_params = function
  | S ->
    "const double *in_re, const double *in_im, double *out_re, double *out_im, double *plane, \
     const double *tw, const size_t *rb, size_t tile"
  | V ->
    "const double *in_re, const double *in_im, double *out_re, double *out_im, double *plane_re, \
     double *plane_im, const double *tw, const size_t *rb, size_t tile, size_t K, size_t blk, \
     size_t gp, size_t g0, size_t g1"
;;

(* the cells: every natural-class ZTURN-T cell (R0 % VW == 0: the S ingest
   writes whole blocks per column) *)
let cells ~(vw : int) () : (int * int list) list =
  List.filter (fun (n, ch) -> Ztt_drivers.natural_ok ~vw n ch) (Ztt_drivers.cells ~vw ())
;;

let emit_driver ~(isa : Isa.t) ~(fam : fam) (n : int) (ch : int list) ~(bwd : bool) : string =
  let g = Ztt_drivers.geom n ch in
  let r = Array.of_list ch in
  let k = Array.length r in
  let sc = scale ~isa fam in
  let body base radix = Cascade_z.ztt_body_name ~fam:(fam_tag fam) ~base ~radix ~bwd in
  let b = Buffer.create 4096 in
  let off = Array.make k 0 in
  for s = 2 to k - 1 do
    off.(s) <- off.(s - 1) + g.Ztt_drivers.twd.(s - 1)
  done;
  let rl s = r.(s) * g.Ztt_drivers.l.(s) in
  (* the group pitch in doubles: the block plane 2 per complex, a lane-batched plane VW *)
  let pitch s = (if fam = S then 2 else sc) * rl s in
  (* mid s (re base, im base): S ignores the im base (one block plane) *)
  let planes_decl, mid, last =
    match fam with
    | S ->
      ( "    double *W = plane;   /* the block interior; the last stage writes the split planes */\n"
      , (fun s (base, _) ->
          Printf.sprintf
            "%s(%s, %s, tw + %d, (size_t)%d, (size_t)%d);"
            (body "tmg" r.(s))
            base
            base
            off.(s)
            g.Ztt_drivers.l.(s)
            g.Ztt_drivers.l.(s))
      , fun s ->
          Printf.sprintf
            "    %s(W, out_re, out_im, tw + %d, (size_t)%d, (size_t)%d, (size_t)%d);\n"
            (body "tlfi" r.(s))
            off.(s)
            g.Ztt_drivers.l.(s)
            g.Ztt_drivers.l.(s)
            g.Ztt_drivers.l.(s) )
    | V ->
      ( "    double *W0 = plane_re, *Wi0 = plane_im;   /* the block's lane-group planes: blk of them at pitch gp */\n"
      , (fun s (base, bi) ->
          Printf.sprintf
            "%s(%s, %s, %s, %s, tw + %d, (size_t)%d, (size_t)%d);"
            (body "tmg" r.(s))
            base
            bi
            base
            bi
            off.(s)
            g.Ztt_drivers.l.(s)
            g.Ztt_drivers.l.(s))
      , fun s ->
          Printf.sprintf
            "    %s(W0, Wi0, go_re, go_im, tw + %d, (size_t)%d, (size_t)%d, K, nb, gp, (size_t)%d);\n"
            (body "tlfi" r.(s))
            off.(s)
            g.Ztt_drivers.l.(s)
            g.Ztt_drivers.l.(s)
            g.Ztt_drivers.l.(s) )
  in
  (* V: the whole pipeline per lane group of VW lanes (the driver's loop);
     the body below is written once and indented one level inside it *)
  let ind = ref (if fam = V then "    " else "") in
  let add s =
    if fam = V
    then (
      let parts = String.split_on_char '\n' s in
      let n = List.length parts in
      List.iteri
        (fun i l ->
           if l <> "" then Buffer.add_string b (!ind ^ l);
           if i < n - 1 then Buffer.add_char b '\n')
        parts)
    else Buffer.add_string b s
  in
  let add_raw = Buffer.add_string b in
  add_raw (Printf.sprintf "__attribute__((target(\"%s\")))\n" isa.Isa.target_attr);
  add_raw (Printf.sprintf "void %s(%s)\n{\n" (driver_name ~isa:isa.Isa.name ~fam n ch ~bwd) (driver_params fam));
  add_raw planes_decl;
  if fam = V
  then
    add_raw
      (Printf.sprintf
         "    /* the lane-group BLOCKS: blk groups of VW = %d lanes per block over the lane range\n\
         \     * [g0, g1) at element pitch K; the ingest and the terminator walk the block's groups\n\
         \     * per row visit (the blocked edges), the mids run per group on its own plane */\n\
         \    for (size_t gl = g0; gl < g1; gl += %d * blk)\n\
         \    {\n\
         \        const size_t nb = ((g1 - gl) / %d < blk) ? (g1 - gl) / %d : blk;\n\
         \        const double *gi_re = in_re + gl, *gi_im = in_im + gl;\n\
         \        double *go_re = out_re + gl, *go_im = out_im + gl;\n"
         isa.Isa.vec_width
         isa.Isa.vec_width
         isa.Isa.vec_width
         isa.Isa.vec_width);
  (* ingest: (in, W, rb, Ls = N/R0, [ep,] count = N/R0) *)
  add
    (match fam with
     | S ->
       Printf.sprintf
         "    %s(in_re, in_im, W, rb, (size_t)%d, (size_t)%d);\n"
         (body "t0tp" r.(0))
         g.Ztt_drivers.ncol
         g.Ztt_drivers.ncol
     | V ->
       Printf.sprintf
         "    %s(gi_re, gi_im, W0, Wi0, rb, (size_t)%d, K, nb, gp, (size_t)%d);\n"
         (body "t0tp" r.(0))
         g.Ztt_drivers.ncol
         g.Ztt_drivers.ncol);
  (* V: the mids per group of the block, on the group's own plane pair *)
  if fam = V
  then (
    add_raw
      "        for (size_t gb = 0; gb < nb; gb++)\n\
      \        {\n\
      \            double *W = W0 + gb * gp, *Wi = Wi0 + gb * gp;\n";
    ind := "        ");
  if k >= 3
  then (
    (* IN-TILE: the mids with R*L <= tile, per tile, a prefix of the mids *)
    add "    if (tile)\n    {\n";
    add (Printf.sprintf "        const size_t ntile = (size_t)%d / tile;\n" n);
    add "#pragma GCC unroll 1\n        for (size_t t = 0; t < ntile; t++)\n        {\n";
    add
      (match fam with
       | S -> "            double *B = W + t * tile * 2;\n"
       | V -> Printf.sprintf "            double *B = W + t * tile * %d, *Bi = Wi + t * tile * %d;\n" sc sc);
    for s = 1 to k - 2 do
      add
        (Printf.sprintf
           "            if ((size_t)%d <= tile)   /* stage %d: R*L = %d */\n            {\n"
           (rl s)
           s
           (rl s));
      add "#pragma GCC unroll 1\n";
      add
        (Printf.sprintf
           "                for (size_t g = 0; g < tile / (size_t)%d; g++)\n                    %s\n            }\n"
           (rl s)
           (mid
              s
              ( Printf.sprintf "B + g * (size_t)%d" (pitch s)
              , Printf.sprintf "Bi + g * (size_t)%d" (pitch s) )))
    done;
    add "        }\n    }\n")
  else add "    (void)tile;   /* no mids: nothing to tile */\n";
  (* CROSS-TILE: the mids with R*L > tile, whole plane *)
  for s = 1 to k - 2 do
    add (Printf.sprintf "    if ((size_t)%d > tile)   /* stage %d: R*L = %d */\n    {\n" (rl s) s (rl s));
    if g.Ztt_drivers.gs.(s) = 1
    then add (Printf.sprintf "        %s\n" (mid s ("W", "Wi")))
    else
      add
        (Printf.sprintf
           "#pragma GCC unroll 1\n        for (size_t g = 0; g < (size_t)%d; g++)\n            %s\n"
           g.Ztt_drivers.gs.(s)
           (mid
              s
              ( Printf.sprintf "W + g * (size_t)%d" (pitch s)
              , Printf.sprintf "Wi + g * (size_t)%d" (pitch s) )));
    add "    }\n"
  done;
  if fam = V
  then (
    ind := "    ";
    add_raw "        }\n");
  let s = k - 1 in
  assert (g.Ztt_drivers.gs.(s) = 1);
  add (last s);
  if fam = V then add_raw "    }\n";
  add_raw "}\n\n";
  (* S: the CONTIGUOUS batch twin, transform-major around the whole pipeline *)
  if fam = S
  then
    add
      (Printf.sprintf
         "__attribute__((target(\"%s\")))\n\
          void %s_batch(%s, size_t K)\n\
          {\n\
         \    for (size_t t = 0; t < K; t++)\n\
         \        %s(in_re + t * %d, in_im + t * %d, out_re + t * %d, out_im + t * %d, plane, tw, rb, tile);\n\
          }\n\n"
         isa.Isa.target_attr
         (driver_name ~isa:isa.Isa.name ~fam n ch ~bwd)
         (driver_params S)
         (driver_name ~isa:isa.Isa.name ~fam n ch ~bwd)
         n
         n
         n
         n);
  Buffer.contents b
;;

let file_name ~(isa : string) ~(fam : fam) (n : int) : string =
  Printf.sprintf "%s_drivers_%s_%d.c" (fam_tag fam) isa n
;;

(* one TU per (family, N): the family's 16 bodies (4 kinds x radix 4/8 x two
   directions, byte-identical to the per-kind codelets of codelets/split/ztt)
   and {fwd, bwd} drivers for every chain of N *)
let emit_file ~(isa : Isa.t) ~(uarch : Uarch.t) ~(fam : fam) (n : int) : string =
  let vw = isa.Isa.vec_width in
  let b = Buffer.create (1 lsl 18) in
  let add = Buffer.add_string b in
  let chains = List.filter_map (fun (m, ch) -> if m = n then Some ch else None) (cells ~vw ()) in
  add
    (Printf.sprintf
       "/* Auto-generated by vfft_v2 — zturn-ts FUSED CODELETS, family %s, N = %d (zts_drivers.ml).\n\
       \ * The SPLIT library's ZTURN-T: one whole-transform function per pow2 cell with the\n\
       \ * stage kernels of codelets/split/ztt/%s INLINED and literal trip counts. The stage\n\
       \ * kernels are the product; a fused codelet cannot be recombined (ZTURN-T's ruling).\n\
       \ * %d cells. ABI: %s\n\
       \ * Generated by: emit_zts_drivers.exe --isa %s --uarch %s --split <dir> */\n"
       (fam_tag fam)
       n
       isa.Isa.name
       (List.length chains)
       (match fam with
        | S ->
          "(in_re, in_im, out_re, out_im, plane, tw, rb, tile), + a _batch twin (..., K) for K contiguous transforms"
        | V -> "(in_re, in_im, out_re, out_im, plane_re, plane_im, tw, rb, tile, K, blk, gp, g0, g1), lane-major K a multiple of VW: the pipeline per BLOCK of blk lane groups (planes at pitch gp) over the lane range [g0, g1)")
       isa.Isa.name
       uarch.Uarch.name);
  add "#include <immintrin.h>\n#include <stddef.h>\n\n";
  List.iter
    (fun (base, radix, bwd) ->
       let kind = kind_name ~fam base ~bwd in
       add
         (Printf.sprintf
            "/* ---- %s radix %d %s (as in radix%d_%s_%s%s_%s.c) ---- */\n"
            kind
            radix
            (if bwd then "bwd" else "fwd")
            radix
            (fam_tag fam)
            base
            (if bwd then "_bwd" else "")
            isa.Isa.name);
       add (Cascade_z.emit_codelet ~body_only:true ~store_on_compute:false ~kind ~radix ~isa ~uarch))
    (List.concat_map
       (fun bwd -> List.map (fun (k, r) -> k, r, bwd) [ "t0tp", 4; "t0tp", 8; "tmg", 4; "tmg", 8; "tlf", 4; "tlf", 8; "tlfi", 4; "tlfi", 8 ])
       [ false; true ]);
  List.iter
    (fun ch ->
       add (Printf.sprintf "/* ==== N=%d chain %s ==== */\n" n (String.concat "." (List.map string_of_int ch)));
       List.iter (fun bwd -> add (emit_driver ~isa ~fam n ch ~bwd)) [ false; true ])
    chains;
  Buffer.contents b
;;

(* write every (family, N) file into dir (the promote rule's action runs in
   generated/fused_codelets) *)
let emit_all ~(isa : Isa.t) ~(uarch : Uarch.t) ~(dir : string) : unit =
  List.iter
    (fun fam ->
       List.iter
         (fun n ->
            let path = Filename.concat dir (file_name ~isa:isa.Isa.name ~fam n) in
            let oc = open_out_bin path in
            output_string oc (emit_file ~isa ~uarch ~fam n);
            close_out oc)
         (Ztt_drivers.sizes ()))
    [ S; V ]
;;
