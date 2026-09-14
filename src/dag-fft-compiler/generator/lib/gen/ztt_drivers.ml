(* ztt_drivers.ml — ZTURN-T: the CELL LIST and the FUSED DRIVER TU.
 *
 * ZTURN-T (docs/design/zturn_t_ship_plan.md; the probe's CONTRACT.md) is the
 * run-contiguous DIT arrangement: ingest t0tp (natural packed z -> plane runs
 * through rb[]), mids tmg (in place, column-varying pre-twiddle, cursor reset
 * per group), last tlf (REINT packed natural out). Per cell (N, chain) the
 * plan binds ONE fused driver — the whole transform as one function with zero
 * calls: the three kind bodies inlined with LITERAL trip counts and the
 * twiddle cursor CARRIED stage to stage in a register (the probe's F5, the
 * form that measured 9-12% at N=128 over the per-stage calls,
 * cascade_stage_fusion.md §2).
 *
 * FUSED CODELETS (owner's ruling 2026-09-14): what this module emits is the
 * pow2 solution's executable form and ONLY that — one whole-transform function
 * per cell with the stage kernels inlined. The stage kernels themselves
 * (codelets/zil/avx2/boundary_split) are the product every other solution
 * composes from; a fused codelet cannot be recombined. They live in
 * generator/generated/fused_codelets/ (README.md there).
 *
 * WHAT THIS MODULE EMITS (ONE FILE PER (family, N) since 2026-09-14 —
 * ztt_drivers_<isa>_<N>.c for the natural order, zttp_drivers_<isa>_<N>.c for
 * the PLAIN = scrambled order, emit_all; the single TU compiled 47 minutes on
 * one thread, the split recompiles only a changed family's files, in parallel):
 *   - the kind bodies the family's drivers inline, static always_inline,
 *     re-emitted by Cascade_z.emit_codelet ~body_only:true (byte-identical to
 *     the bodies inside the per-kind codelets, so fused == unfused is gate-able
 *     bitwise); natural: t0tp/tmg/tlf/tlfi x two directions; plain: t0d/tmgd/
 *     tld forward, tld backward, and the natural tmg/tlf backward bodies;
 *   - for every natural cell: {fwd, bwd} x {dest, plane} drivers; for every
 *     plain cell: {fwd, bwd} (emit_plain_driver below — no plane, no rb).
 *   SIZE: ~4 KB of source per driver, 223 cells x 6 drivers; only one driver is
 *   live per plan, but the library's text carries them all.
 *       dest : the pipeline runs IN THE DESTINATION (zin != zout, zout
 *              64-B aligned); the plane argument is unused.
 *       plane: the pipeline runs in the plan's scratch plane (in place, or
 *              an unaligned destination); the last stage writes zout — as
 *              tlfi, the IN-PLACE terminator (tlf with its output streams
 *              prefetched: zout's lines went cold under the plane, and each
 *              store waited on its fill, +17..25%; zturn_t_ship_plan.md 9).
 *   The registry (bin/emit_ztt_registry.ml) lists the same cells, so
 *   "exists" and "reachable" cannot diverge.
 *   TILING (2026-09-09): every driver takes the tile width as a RUNTIME
 *   argument (the mids with R*L <= tile run per tile, the rest sweep the
 *   plane; the hot loops inside a group stay literal) — one driver per cell,
 *   the width a raced plan parameter, as the cascade's tcut.
 *
 * CELLS: every ordered {4,8} chain with product N, nf >= 2, R0 % 4 == 0 and
 * (N / R0) % 4 == 0 (CONTRACT.md §1 — every plane address a kernel touches
 * is a whole 64-B block), for 16 <= N <= 262144 (the cascade's ceiling, S4
 * 2026-09-09). Up to RL = 16384 the create expands its streams from the baked
 * quarter-wave by an index shift; above it by the two-level product (ztt.h) (zt_bake.c: "table resolution
 * refusal"). *)

let max_n = 262144
let max_nf = 7 (* VFFT_ZSPLIT_MAX_NF *)

(* every ordered {4,8} chain with product n, nf >= 2, (n / r0) mod 4 = 0;
   the order is the enumeration order (4 before 8 at every position) *)
let chains_of (n : int) : int list list =
  let rec go prod acc depth =
    if prod = n
    then if List.length acc >= 2 then [ List.rev acc ] else []
    else if prod > n || depth >= max_nf
    then []
    else go (prod * 4) (4 :: acc) (depth + 1) @ go (prod * 8) (8 :: acc) (depth + 1)
  in
  List.filter (fun ch -> n / List.hd ch mod 4 = 0) (go 1 [] 0)
;;

let cells () : (int * int list) list =
  let rec pow2 n acc = if n > max_n then List.rev acc else pow2 (2 * n) (n :: acc) in
  List.concat_map (fun n -> List.map (fun ch -> n, ch) (chains_of n)) (pow2 16 [])
;;

(* per-stage geometry (CONTRACT.md §1): L[1] = R0, L[s+1] = L[s]*R[s],
   Gs[s] = N/(R[s]*L[s]); stream doubles per stage = 2*(R-1)*L (16(R-1)L B) *)
type geom =
  { ncol : int
  ; l : int array
  ; gs : int array
  ; twd : int array
  }

let geom (n : int) (ch : int list) : geom =
  let k = List.length ch in
  let r = Array.of_list ch in
  let l = Array.make k 0
  and gs = Array.make k 0
  and twd = Array.make k 0 in
  l.(1) <- r.(0);
  for s = 1 to k - 1 do
    let rl = l.(s) * r.(s) in
    gs.(s) <- n / rl;
    twd.(s) <- 2 * (r.(s) - 1) * l.(s);
    if s + 1 < k then l.(s + 1) <- rl
  done;
  { ncol = n / r.(0); l; gs; twd }
;;

let tag (n : int) (ch : int list) : string =
  string_of_int n ^ "_" ^ String.concat "_" (List.map string_of_int ch)
;;

let driver_name ~(isa : string) (n : int) (ch : int list) ~(bwd : bool) ~(dest : bool) : string =
  Printf.sprintf
    "ztt_%s_%s_%s_%s"
    (tag n ch)
    (if bwd then "bwd" else "fwd")
    (if dest then "dest" else "plane")
    isa
;;

(* the driver ABI, shared with the registry: (zin, zout, plane, tw, rb, tile).
   tile = the TILE WIDTH in complexes (0 = untiled): the mid stages whose run
   length R*L <= tile run PER TILE — a contiguous plane span of `tile`
   complexes holds WHOLE groups of every such stage — the rest sweep the plane
   (docs/design/zturn_t_2048plus_plan.md step 2). A RACED plan parameter
   (il_tw=, validated by vfft_ztt_tile_legal), never a rule; the tile changes
   group ORDER only, so every width is bitwise the untiled result. *)
let driver_params =
  "const double *zin, double *zout, double *plane, const double *tw, const size_t *rb, size_t tile"
;;

let emit_driver ~(isa : Isa.t) (n : int) (ch : int list) ~(bwd : bool) ~(dest : bool) : string =
  let g = geom n ch in
  let r = Array.of_list ch in
  let k = Array.length r in
  let body base radix = Cascade_z.ztt_body_name ~base ~radix ~bwd in
  let b = Buffer.create 4096 in
  let add = Buffer.add_string b in
  add (Printf.sprintf "__attribute__((target(\"%s\")))\n" isa.Isa.target_attr);
  add
    (Printf.sprintf
       "void %s(%s)\n{\n"
       (driver_name ~isa:isa.Isa.name n ch ~bwd ~dest)
       driver_params);
  if dest
  then add "    double *W = zout;   /* dest: the whole pipeline runs in the destination */\n    (void)plane;\n"
  else add "    double *W = plane;  /* plane: the plan's scratch; the last stage writes zout */\n";
  (* ingest: (zin, plane, rb, Ls = N/R0, count = N/R0)          CONTRACT 7.1 *)
  add
    (Printf.sprintf
       "    %s(zin, W, rb, (size_t)%d, (size_t)%d);\n"
       (body "t0tp" r.(0))
       g.ncol
       g.ncol);
  (* stage s's stream starts at a LITERAL offset into the plan's ONE contiguous
     stream (stage order) — a tile re-enters every stage from its stream base *)
  let off = Array.make k 0 in
  for s = 2 to k - 1 do
    off.(s) <- off.(s - 1) + g.twd.(s - 1)
  done;
  let rl s = r.(s) * g.l.(s) in
  let pitch s = 2 * rl s in
  let mid s base =
    Printf.sprintf
      "%s(%s, %s, tw + %d, (size_t)%d, (size_t)%d);"
      (body "tmg" r.(s))
      base
      base
      off.(s)
      g.l.(s)
      g.l.(s)
  in
  if k >= 3
  then (
    (* IN-TILE: the mids with R*L <= tile (a PREFIX of the mids: R*L grows
       with s), per tile, stage-major inside the tile, tile/(R*L) groups each *)
    add "    if (tile)\n    {\n";
    add (Printf.sprintf "        const size_t ntile = (size_t)%d / tile;\n" n);
    add "#pragma GCC unroll 1\n        for (size_t t = 0; t < ntile; t++)\n        {\n";
    add "            double *B = W + t * tile * 2;\n";
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
           (mid s (Printf.sprintf "B + g * (size_t)%d" (pitch s))))
    done;
    add "        }\n    }\n")
  else add "    (void)tile;   /* no mids: nothing to tile */\n";
  (* CROSS-TILE: the mids with R*L > tile (every mid when untiled), whole plane *)
  for s = 1 to k - 2 do
    add (Printf.sprintf "    if ((size_t)%d > tile)   /* stage %d: R*L = %d */\n    {\n" (rl s) s (rl s));
    if g.gs.(s) = 1
    then add (Printf.sprintf "        %s\n" (mid s "W"))
    else
      add
        (Printf.sprintf
           "#pragma GCC unroll 1\n        for (size_t g = 0; g < (size_t)%d; g++)\n            %s\n"
           g.gs.(s)
           (mid s (Printf.sprintf "W + g * (size_t)%d" (pitch s))));
    add "    }\n"
  done;
  (* last: (W, zout, tw, Ls = L, OLs = L, count = L); Gs == 1 for the
     terminal stage of a K-stage chain — a direct call.       CONTRACT 7.3 *)
  let s = k - 1 in
  assert (g.gs.(s) = 1);
  add
    (Printf.sprintf
       "    %s(W, zout, tw + %d, (size_t)%d, (size_t)%d, (size_t)%d);\n"
       (body (if dest then "tlf" else "tlfi") r.(s))
       off.(s)
       g.l.(s)
       g.l.(s)
       g.l.(s));
  add "}\n\n";
  Buffer.contents b
;;

(* ═══ ZTURN-T PLAIN = the scrambled class (docs/design/ztt_scrambled_design.md,
   2026-09-13): Sande-Tukey IN PLACE on the chain.
     Len_0 = N, Len_{s+1} = Len_s / R_s; stage s: Ls = count = Len_{s+1},
     Gs = N / Len_s, POST-twiddle w_{Len_s}^(p*b) — the stream record shape and
     byte total of the natural schedule on the reversed chain.
   Forward: t0d (stage 0, the ONE sweep: interleaved legs at stride N/R0 from
   zin, block-split leg-major into zout), tmgd mids in place, tld last (adjacent
   legs, twiddle-free, interleaved stores into the same span). The stages with
   Len_s > tile sweep FIRST (a prefix — Len shrinks with s); the stages with
   Len_s <= tile, the last one always, run per block of `tile` complexes
   AFTER them (the mirror of the natural loop, whose tiled stages are the
   prefix). Backward = the stage-by-stage inverse in reverse order: per block
   tldb then tmgb (PRE conj + IDFT) for the tiled stages, then the tmgb sweeps,
   then tlfb (in place, natural interleaved out). One driver per direction: no
   plane, no rb[], zin == zout is the same driver (every stage loads its whole
   column quad before storing it). ═══ *)
let plain_params = "const double *zin, double *zout, const double *tw, size_t tile"

let plain_driver_name ~(isa : string) (n : int) (ch : int list) ~(bwd : bool) : string =
  Printf.sprintf "zttp_%s_%s_%s" (tag n ch) (if bwd then "bwd" else "fwd") isa
;;

(* per-stage geometry of the plain schedule: len.(s) = prod_{u >= s} R[u]
   (len.(k) = 1); stream doubles per stage s <= k-2 = 2*(R_s - 1)*len.(s+1),
   offsets prefix-summed in stage order; the last stage carries no stream *)
type geom_plain =
  { len : int array
  ; poff : int array
  }

let geom_plain (n : int) (ch : int list) : geom_plain =
  let k = List.length ch in
  let r = Array.of_list ch in
  let len = Array.make (k + 1) 1 in
  for s = k - 1 downto 0 do
    len.(s) <- len.(s + 1) * r.(s)
  done;
  assert (len.(0) = n);
  let poff = Array.make k 0 in
  for s = 1 to k - 1 do
    poff.(s) <- poff.(s - 1) + (2 * (r.(s - 1) - 1) * len.(s))
  done;
  { len; poff }
;;

let emit_plain_driver ~(isa : Isa.t) (n : int) (ch : int list) ~(bwd : bool) : string =
  let g = geom_plain n ch in
  let r = Array.of_list ch in
  let k = Array.length r in
  let body base radix ~bwd = Cascade_z.ztt_body_name ~base ~radix ~bwd in
  let b = Buffer.create 4096 in
  let add = Buffer.add_string b in
  add (Printf.sprintf "__attribute__((target(\"%s\")))\n" isa.Isa.target_attr);
  add
    (Printf.sprintf
       "void %s(%s)\n{\n"
       (plain_driver_name ~isa:isa.Isa.name n ch ~bwd)
       plain_params);
  (* stage s mid call (fwd: tmgd, bwd: tmgb) at a group base *)
  let mid s base =
    Printf.sprintf
      "%s(%s, %s, tw + %d, (size_t)%d, (size_t)%d);"
      (body (if bwd then "tmg" else "tmgd") r.(s) ~bwd)
      base
      base
      g.poff.(s)
      g.len.(s + 1)
      g.len.(s + 1)
  in
  let pitch s = 2 * g.len.(s) in
  let sweep s =
    add
      (Printf.sprintf
         "    if ((size_t)%d > tile)   /* stage %d: Len = %d, sweeps */\n    {\n"
         g.len.(s)
         s
         g.len.(s));
    if n / g.len.(s) = 1
    then add (Printf.sprintf "        %s\n" (mid s "zout"))
    else
      add
        (Printf.sprintf
           "#pragma GCC unroll 1\n        for (size_t g = 0; g < (size_t)%d; g++)\n            %s\n"
           (n / g.len.(s))
           (mid s (Printf.sprintf "zout + g * (size_t)%d" (pitch s))));
    add "    }\n"
  in
  let per_block s =
    add
      (Printf.sprintf
         "            if ((size_t)%d <= tile)   /* stage %d: Len = %d */\n            {\n"
         g.len.(s)
         s
         g.len.(s));
    add "#pragma GCC unroll 1\n";
    add
      (Printf.sprintf
         "                for (size_t g = 0; g < tile / (size_t)%d; g++)\n                    %s\n            }\n"
         g.len.(s)
         (mid s (Printf.sprintf "B + g * (size_t)%d" (pitch s))))
  in
  let rlast = r.(k - 1) in
  if not bwd
  then (
    (* stage 0: the one sweep — zin -> zout (in place when equal)  *)
    add
      (Printf.sprintf
         "    %s(zin, zout, tw + 0, (size_t)%d, (size_t)%d);\n"
         (body "t0d" r.(0) ~bwd:false)
         g.len.(1)
         g.len.(1));
    (* the cross-tile mids (every mid when untiled), in stage order *)
    for s = 1 to k - 2 do
      sweep s
    done;
    (* the per-block suffix: the mids with Len <= tile, then the last stage *)
    add "    if (tile)\n    {\n";
    add (Printf.sprintf "        const size_t ntile = (size_t)%d / tile;\n" n);
    add "#pragma GCC unroll 1\n        for (size_t t = 0; t < ntile; t++)\n        {\n";
    add "            double *B = zout + t * tile * 2;\n";
    for s = 1 to k - 2 do
      per_block s
    done;
    add
      (Printf.sprintf
         "            %s(B, B, tw, (size_t)0, tile / (size_t)%d);   /* last stage: Len = %d, in place */\n"
         (body "tld" rlast ~bwd:false)
         rlast
         rlast);
    add "        }\n    }\n    else\n";
    add
      (Printf.sprintf
         "        %s(zout, zout, tw, (size_t)0, (size_t)%d);   /* last stage, untiled: one sweep in place */\n"
         (body "tld" rlast ~bwd:false)
         (n / rlast)))
  else (
    (* the inverse, stage by stage in reverse: per block tldb then the tiled
       mids (highest stage first), then the sweeping mids, then tlfb *)
    add "    if (tile)\n    {\n";
    add (Printf.sprintf "        const size_t ntile = (size_t)%d / tile;\n" n);
    add "#pragma GCC unroll 1\n        for (size_t t = 0; t < ntile; t++)\n        {\n";
    add "            const double *Bi = zin + t * tile * 2;\n";
    add "            double *B = zout + t * tile * 2;\n";
    add
      (Printf.sprintf
         "            %s(Bi, B, tw, (size_t)0, tile / (size_t)%d);   /* inverse of the last stage */\n"
         (body "tld" rlast ~bwd:true)
         rlast);
    for s = k - 2 downto 1 do
      per_block s
    done;
    add "        }\n    }\n    else\n";
    add
      (Printf.sprintf
         "        %s(zin, zout, tw, (size_t)0, (size_t)%d);\n"
         (body "tld" rlast ~bwd:true)
         (n / rlast));
    for s = k - 2 downto 1 do
      sweep s
    done;
    (* inverse of stage 0: tlfb — block-split legs at stride Len_1, PRE conj,
       IDFT, natural interleaved out at the same stride: in place *)
    add
      (Printf.sprintf
         "    %s(zout, zout, tw + 0, (size_t)%d, (size_t)%d, (size_t)%d);\n"
         (body "tlf" r.(0) ~bwd:true)
         g.len.(1)
         g.len.(1)
         g.len.(1)));
  add "}\n\n";
  Buffer.contents b
;;

(* ═══ THE DRIVER FILES (2026-09-14): one TU per (family, N) — 15 sizes x
   {natural, plain} = 30 files — instead of one 3.9 MB TU that compiled 47
   minutes on one thread. A change to one family's kinds recompiles only that
   family's files, in parallel (build.py globs generated/*.c and drops the
   objects of sources that are gone). Each file re-emits the bodies its
   drivers inline (static, byte-identical to the per-kind codelets); the
   natural file also carries the s0t masks its ingest bodies need. ═══ *)
let sizes () : int list =
  let rec pow2 n acc = if n > max_n then List.rev acc else pow2 (2 * n) (n :: acc) in
  pow2 16 []
;;

let file_name ~(isa : string) ~(scr : bool) (n : int) : string =
  Printf.sprintf "%s_drivers_%s_%d.c" (if scr then "zttp" else "ztt") isa n
;;

(* the bodies a family's drivers inline: natural = the four kinds x two
   directions; plain = t0d / tmgd / tld forward, tld backward, and the natural
   tmg / tlf BACKWARD bodies its backward driver walks *)
let bodies ~(scr : bool) : (string * int * bool) list =
  if not scr
  then
    List.concat_map
      (fun bwd ->
         List.map
           (fun (k, r) -> k, r, bwd)
           [ "t0tp", 4; "t0tp", 8; "tmg", 4; "tmg", 8; "tlf", 4; "tlf", 8; "tlfi", 4; "tlfi", 8 ])
      [ false; true ]
  else
    [ "t0d", 4, false; "t0d", 8, false; "tmgd", 4, false; "tmgd", 8, false
    ; "tld", 4, false; "tld", 8, false; "tld", 4, true; "tld", 8, true
    ; "tmg", 4, true; "tmg", 8, true; "tlf", 4, true; "tlf", 8, true ]
;;

let emit_file ~(isa : Isa.t) ~(uarch : Uarch.t) ~(scr : bool) (n : int) : string =
  let b = Buffer.create (1 lsl 18) in
  let add = Buffer.add_string b in
  let cells = List.filter (fun (m, _) -> m = n) (cells ()) in
  add
    (Printf.sprintf
       "/* Auto-generated by vfft_v2 — ZTURN-T FUSED CODELETS, %s, N = %d (ztt_drivers.ml).\n\
       \ * FUSED CODELET = one whole-transform function per pow2 cell with the stage\n\
       \ * kernels of codelets/zil/avx2/boundary_split INLINED and literal trip counts:\n\
       \ * the pow2 ZTURN-T solution's executable form, and only that (owner's ruling\n\
       \ * 2026-09-14, README.md in this directory). The stage kernels are the product\n\
       \ * every other solution composes from; a fused codelet cannot be recombined.\n\
       \ * %d cells; the kind bodies below are inlined with LITERAL trip counts\n\
       \ * (docs/design/cascade_stage_fusion.md). %s\n\
       \ * Generated by: emit_ztt_drivers.exe --isa %s --uarch %s --split <dir> */\n"
       (if scr then "PLAIN = scrambled order (ztt_scrambled_design.md)" else "NATURAL order")
       n
       (List.length cells)
       (if scr
        then
          "ABI: (zin, zout, tw, tile) — no plane, no rb; the stages with\n\
           \ * Len_s <= tile and the last stage run per block; zin == zout is the same driver."
        else
          "ABI: (zin, zout, plane, tw, rb, tile) — tw = the plan's ONE\n\
           \ * contiguous stream (stage order), rb = the run-base table (CONTRACT.md 5), tile =\n\
           \ * the tile width in complexes (0 = untiled; the mids with R*L <= tile run per tile).")
       isa.Isa.name
       uarch.Uarch.name);
  add "#include <immintrin.h>\n#include <stddef.h>\n\n";
  if not scr
  then (
    add (Isa.im_mask_decl isa "_zs0t_mim" ^ "   /* x(-i): the forward quarter-turn */\n");
    add (Isa.re_mask_decl isa "_zs0t_pim" ^ "   /* x(+i): the backward quarter-turn */\n");
    add
      "static const __m256d _zs0t_rh = { 0.70710678118654752440, 0.70710678118654752440, \
       0.70710678118654752440, 0.70710678118654752440 };  /* 1/sqrt2: |W8^1| */\n\n");
  List.iter
    (fun (kind, radix, bwd) ->
       add
         (Printf.sprintf
            "/* ---- %s radix %d %s (as in radix%d_z_%s%s_avx2.c) ---- */\n"
            kind
            radix
            (if bwd then "bwd" else "fwd")
            radix
            kind
            (if bwd then "_bwd" else ""));
       add
         (Cascade_z.emit_codelet
            ~body_only:true
            ~store_on_compute:false
            ~kind:(kind ^ if bwd then "b" else "")
            ~radix
            ~r0:None
            ~sink_stores:false
            ~sched:None
            ~isa
            ~uarch))
    (bodies ~scr);
  List.iter
    (fun (m, ch) ->
       add
         (Printf.sprintf
            "/* ==== N=%d chain %s ==== */\n"
            m
            (String.concat "." (List.map string_of_int ch)));
       if scr
       then List.iter (fun bwd -> add (emit_plain_driver ~isa m ch ~bwd)) [ false; true ]
       else
         List.iter
           (fun (bwd, dest) -> add (emit_driver ~isa m ch ~bwd ~dest))
           [ false, true; false, false; true, true; true, false ])
    cells;
  Buffer.contents b
;;

(* write every file into dir (the dune promote rule's action runs in generated/) *)
let emit_all ~(isa : Isa.t) ~(uarch : Uarch.t) ~(dir : string) : unit =
  List.iter
    (fun scr ->
       List.iter
         (fun n ->
            let path = Filename.concat dir (file_name ~isa:isa.Isa.name ~scr n) in
            let oc = open_out_bin path in
            output_string oc (emit_file ~isa ~uarch ~scr n);
            close_out oc)
         (sizes ()))
    [ false; true ]
;;
