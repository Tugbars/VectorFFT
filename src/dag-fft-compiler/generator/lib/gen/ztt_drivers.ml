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
 * WHAT THIS MODULE EMITS (one TU per ISA, --ztt-drivers):
 *   - the six kind bodies x two directions, static always_inline, re-emitted
 *     by Cascade_z.emit_codelet ~body_only:true (byte-identical to the bodies
 *     inside the per-kind codelets, so fused == unfused is gate-able bitwise);
 *   - for every cell: {fwd, bwd} x {dest, plane} drivers.
 *       dest : the pipeline runs IN THE DESTINATION (zin != zout, zout
 *              64-B aligned); the plane argument is unused.
 *       plane: the pipeline runs in the plan's scratch plane (in place, or
 *              an unaligned destination); the last stage writes zout.
 *   The registry (bin/emit_ztt_registry.ml) lists the same cells, so
 *   "exists" and "reachable" cannot diverge.
 *   TILING (2026-09-09): every driver takes the tile width as a RUNTIME
 *   argument (the mids with R*L <= tile run per tile, the rest sweep the
 *   plane; the hot loops inside a group stay literal) — one driver per cell,
 *   the width a raced plan parameter, as the cascade's tcut.
 *
 * CELLS: every ordered {4,8} chain with product N, nf >= 2, R0 % 4 == 0 and
 * (N / R0) % 4 == 0 (CONTRACT.md §1 — every plane address a kernel touches
 * is a whole 64-B block), for 16 <= N <= 16384. The ceiling is STRUCTURAL:
 * the create expands its streams from a baked quarter-wave at M = 16384 by an
 * index shift, which cannot resolve RL > M (zt_bake.c: "table resolution
 * refusal"). *)

let max_n = 16384
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
       (body "tlf" r.(s))
       off.(s)
       g.l.(s)
       g.l.(s)
       g.l.(s));
  add "}\n\n";
  Buffer.contents b
;;

let emit_tu ~(isa : Isa.t) ~(uarch : Uarch.t) : string =
  let b = Buffer.create (1 lsl 20) in
  let add = Buffer.add_string b in
  let cells = cells () in
  add
    (Printf.sprintf
       "/* Auto-generated by vfft_v2 — ZTURN-T FUSED DRIVERS (ztt_drivers.ml).\n\
       \ * One function per (N, chain, direction, buffer mode): the t0tp / tmg / tlf\n\
       \ * bodies below inlined with LITERAL trip counts and the twiddle cursor carried\n\
       \ * in a register across stages (docs/design/cascade_stage_fusion.md).\n\
       \ * %d cells x {fwd, bwd} x {dest, plane} = %d drivers. ABI: (zin, zout, plane,\n\
       \ * tw, rb, tile) — tw = the plan's ONE contiguous stream (stage order), rb = the\n\
       \ * run-base table (CONTRACT.md 5), tile = the tile width in complexes (0 =\n\
       \ * untiled; the mids with R*L <= tile run per tile). Generated by: gen_radix.exe 4 --ztt-drivers\n\
       \ * --isa %s --uarch %s --emit-c */\n"
       (List.length cells)
       (4 * List.length cells)
       isa.Isa.name
       uarch.Uarch.name);
  add "#include <immintrin.h>\n#include <stddef.h>\n\n";
  add (Isa.im_mask_decl isa "_zs0t_mim" ^ "   /* x(-i): the forward quarter-turn */\n");
  add (Isa.re_mask_decl isa "_zs0t_pim" ^ "   /* x(+i): the backward quarter-turn */\n");
  add
    "static const __m256d _zs0t_rh = { 0.70710678118654752440, 0.70710678118654752440, \
     0.70710678118654752440, 0.70710678118654752440 };  /* 1/sqrt2: |W8^1| */\n\n";
  (* the twelve bodies, byte-identical to the per-kind codelets' *)
  List.iter
    (fun bwd ->
       List.iter
         (fun (kind, radix) ->
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
         [ "t0tp", 4; "t0tp", 8; "tmg", 4; "tmg", 8; "tlf", 4; "tlf", 8 ])
    [ false; true ];
  (* the drivers *)
  List.iter
    (fun (n, ch) ->
       add (Printf.sprintf "/* ==== N=%d chain %s ==== */\n" n (String.concat "." (List.map string_of_int ch)));
       List.iter
         (fun (bwd, dest) -> add (emit_driver ~isa n ch ~bwd ~dest))
         [ false, true; false, false; true, true; true, false ])
    cells;
  Buffer.contents b
;;
