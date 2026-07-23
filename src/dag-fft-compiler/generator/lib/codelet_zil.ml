(* codelet_zil.ml — TRUE interleaved-native (z-layout) codelet emitter.
 *
 * Tier-2 of the IL-native workstream (docs/roadmap/il_native_design.md,
 * docs/research/mkl_il_512_anatomy.md). Unlike the split codelets +
 * il_in/il_out boundary lattices (the DELETED il_derive population and the
 * K=1 boundary twins), these codelets keep every value INTERLEAVED
 * (2 complex per %ymm, [re0,im0,re1,im1]) end-to-end — no split re/im planes,
 * no boundary conversion. This is FFTW's n1fv / MKL's zzd2 model.
 *
 * The existing emitter (codelet_oop.ml + emit_c.ml + algsimp) is a REAL-valued
 * (split) backend: the `expr` type is Const/Load/Add/Sub/Mul over reals and the
 * DAG carries CmulRe/CmulIm nodes. That backend cannot be "re-rendered"
 * interleaved — it has already split complex into real subtrees. So this is a
 * separate, small complex-vector backend for the pow-2 radices (which cover
 * every cell we ship: 512 = 8/16/32/64, etc.).
 *
 * M1 (this file, first cut): radix-8 n1 leaf (twiddle-free), forward, matching
 * the bit-exact hand oracle in build_tuned/benches/il_r8_m1_race.c (r8_z).
 * M2+: radix 4/16/32/64, the t2 twiddle variant (VTW2 table + BYTW2), backward.
 *
 * IL OP-SELECTION RULES (the body is DESIGNED from the IL arsenal, not
 * transliterated from the split DAG — user directive 2026-07-24; MKL census
 * docs/research/mkl_il_512_anatomy.md is the reference op shape):
 *   1. add/sub          -> vaddpd/vsubpd directly on packed complex.
 *   2. PAIRED a±i*b     -> ONE shared rotation (permute 0x5 + xor M_IM), then
 *      add+sub. For sum-AND-difference pairs (every DIT butterfly) this is
 *      op-optimal; vaddsubpd wins nothing here.
 *   3. LONE a+i*b       -> FLIP + vaddsubpd (2 ops, no xor). Appears in the
 *      lone-rotation/conj sites of radix>=16 (M2) and the ±i boundaries of
 *      the t2 combine (MKL: 23 triads). Not present in radix-8 n1.
 *   4. CONST-SCALED combine  out = e ± C*x  -> vfmadd213/vfnmadd213 folding
 *      the scale into the butterfly (MKL's 4-FMA radix-8 shape). Never a
 *      separate vmulpd + add/sub.
 *   5. Twiddle mul (t2, M2)  -> BYTW2 against the VTW2 cos-first table:
 *      fma(cos_v, x, mul(sin_v, FLIP(x))) — 1 data-side shuffle, zero
 *      table-side shuffles/xor (sign folded, cos duplicated at table build).
 *)

(* ═══════════════════════════════════════════════════════════════════
 * COMPLEX-VECTOR IR + recursive DIT builder (radices 4/16/32/64; radix-8
 * keeps the hand-shaped gold body below, bit-gated vs the M1 oracle).
 *
 * Node classes map 1:1 onto the op-selection rules:
 *   In       — z load (2 complex/ymm)
 *   Add/Sub  — vaddpd/vsubpd on packed complex
 *   RotNI    — *(-i): permute 0x5 + xor _M_IM (rule 2; CSE shares pairs)
 *   Fmadd/Fnmadd — c*x ± folded into the butterfly (rule 4)
 *   CTw      — *(c+i·s) general CONSTANT twiddle, BYTW2 shape with emit-time
 *              VLIT constants: FLIP + mul([-s,+s]) + fmadd([c,c]) (rule 5,
 *              constant-folded — no table, no broadcasts at runtime)
 * DIT-2 recursion, natural order in and out; forward sign e^{-2πik/n}.
 * ═══════════════════════════════════════════════════════════════════ *)
type zx =
  | In of int
  | Add of zx * zx
  | Sub of zx * zx
  | RotNI of zx
  | Fmadd of float * zx * zx   (* c*x + e *)
  | Fnmadd of float * zx * zx  (* e - c*x *)
  | CTw of float * float * zx  (* x * (c + i*s), constants *)

let sqh = 0.70710678118654752440

(* forward DIT radix-2 recursion over the point list; returns outputs in
 * natural order. Twiddle classes chosen per k as in the header. *)
let rec dft_z (n : int) (xs : zx array) : zx array =
  if n = 1 then xs
  else begin
    let h = n / 2 in
    let ev = Array.init h (fun i -> xs.(2 * i)) in
    let od = Array.init h (fun i -> xs.((2 * i) + 1)) in
    let e = dft_z h ev and o = dft_z h od in
    let out = Array.make n (In 0) in
    let pi = 4.0 *. atan 1.0 in
    for k = 0 to h - 1 do
      let c = cos (2.0 *. pi *. float_of_int k /. float_of_int n)
      and s = -.sin (2.0 *. pi *. float_of_int k /. float_of_int n) in
      if k = 0 then begin
        out.(k) <- Add (e.(k), o.(k));
        out.(k + h) <- Sub (e.(k), o.(k))
      end
      else if 4 * k = n then begin
        let t = RotNI o.(k) in
        out.(k) <- Add (e.(k), t);
        out.(k + h) <- Sub (e.(k), t)
      end
      else if 8 * k = n then begin
        (* w = (1-i)/sqrt2: fold the scale into the butterfly (rule 4) *)
        let x = Add (o.(k), RotNI o.(k)) in
        out.(k) <- Fmadd (sqh, x, e.(k));
        out.(k + h) <- Fnmadd (sqh, x, e.(k))
      end
      else if 8 * k = 3 * n then begin
        (* w = -(1+i)/sqrt2 = sqh * (rot(x) - x) *)
        let x = Sub (RotNI o.(k), o.(k)) in
        out.(k) <- Fmadd (sqh, x, e.(k));
        out.(k + h) <- Fnmadd (sqh, x, e.(k))
      end
      else begin
        let t = CTw (c, s, o.(k)) in
        out.(k) <- Add (e.(k), t);
        out.(k + h) <- Sub (e.(k), t)
      end
    done;
    out
  end

(* lowering: hash-consed emission (each distinct node once, topological),
 * distinct CTw constants deduped into static VLIT vectors. *)
let emit_dft_body (buf : Buffer.t) ~(ind : string) (radix : int)
    (consts : (string, string * float * float) Hashtbl.t) : unit =
  let outs = dft_z radix (Array.init radix (fun i -> In i)) in
  let memo : (zx, string) Hashtbl.t = Hashtbl.create 128 in
  let ctr = ref 0 in
  let fresh () = incr ctr; Printf.sprintf "z%d" !ctr in
  let cid (c, s) =
    let key = Printf.sprintf "%.17g_%.17g" c s in
    match Hashtbl.find_opt consts key with
    | Some (n, _, _) -> n
    | None ->
        let n = Printf.sprintf "_ZW%d" (Hashtbl.length consts) in
        Hashtbl.add consts key (n, c, s);
        n
  in
  let rec go (e : zx) : string =
    match Hashtbl.find_opt memo e with
    | Some v -> v
    | None ->
        let v =
          match e with
          | In i -> Printf.sprintf "in%d" i
          | Add (a, b) ->
              let a = go a and b = go b and t = fresh () in
              Buffer.add_string buf
                (Printf.sprintf "%s__m256d %s=_mm256_add_pd(%s,%s);\n" ind t a b);
              t
          | Sub (a, b) ->
              let a = go a and b = go b and t = fresh () in
              Buffer.add_string buf
                (Printf.sprintf "%s__m256d %s=_mm256_sub_pd(%s,%s);\n" ind t a b);
              t
          | RotNI a ->
              let a = go a and t = fresh () in
              Buffer.add_string buf
                (Printf.sprintf
                   "%s__m256d %s=_mm256_xor_pd(_mm256_permute_pd(%s,0x5),_M_IM);\n"
                   ind t a);
              t
          | Fmadd (c, x, e') ->
              let x = go x and e' = go e' and t = fresh () in
              Buffer.add_string buf
                (Printf.sprintf
                   "%s__m256d %s=_mm256_fmadd_pd(_mm256_set1_pd(%.17g),%s,%s);\n"
                   ind t c x e');
              t
          | Fnmadd (c, x, e') ->
              let x = go x and e' = go e' and t = fresh () in
              Buffer.add_string buf
                (Printf.sprintf
                   "%s__m256d %s=_mm256_fnmadd_pd(_mm256_set1_pd(%.17g),%s,%s);\n"
                   ind t c x e');
              t
          | CTw (c, s, x) ->
              (* BYTW2-with-constants: FLIP + mul(svec) + fmadd(cvec) *)
              let x = go x in
              let w = cid (c, s) in
              let t = fresh () in
              Buffer.add_string buf
                (Printf.sprintf
                   "%s__m256d %s=_mm256_fmadd_pd(%s_c,%s,_mm256_mul_pd(%s_s,_mm256_permute_pd(%s,0x5)));\n"
                   ind t w x w x);
              t
        in
        Hashtbl.add memo e v;
        v
  in
  Array.iteri
    (fun p e ->
      let v = go e in
      Buffer.add_string buf (Printf.sprintf "%sout%d=%s;\n" ind p v))
    outs

(* radix-8 forward n1, interleaved. `outs.(p)` is the C expression producing
 * output point p from inputs in.(0..7). EXACT translation of the verified
 * split dft8v (mono-64) — see the race oracle. Emitted straight-line; gcc
 * schedules. *)
let r8_body (buf : Buffer.t) ~(ind : string) : unit =
  let p s = Buffer.add_string buf (ind ^ s ^ "\n") in
  p "const __m256d _C = _mm256_set1_pd(0.70710678118654752440);";
  (* t = radix-2 on even/odd 4-groups *)
  p "__m256d t0=_mm256_add_pd(in0,in4), t1=_mm256_sub_pd(in0,in4);";
  p "__m256d t2=_mm256_add_pd(in2,in6), t3=_mm256_sub_pd(in2,in6);";
  p "__m256d E0=_mm256_add_pd(t0,t2),   E2=_mm256_sub_pd(t0,t2);";
  p "__m256d _t3ni=_mm256_xor_pd(_mm256_permute_pd(t3,0x5),_M_IM); /* t3*(-i) */";
  p "__m256d E1=_mm256_add_pd(t1,_t3ni), E3=_mm256_sub_pd(t1,_t3ni);";
  p "__m256d s0=_mm256_add_pd(in1,in5), s1=_mm256_sub_pd(in1,in5);";
  p "__m256d s2=_mm256_add_pd(in3,in7), s3=_mm256_sub_pd(in3,in7);";
  p "__m256d O0=_mm256_add_pd(s0,s2),   O2=_mm256_sub_pd(s0,s2);";
  p "__m256d _s3ni=_mm256_xor_pd(_mm256_permute_pd(s3,0x5),_M_IM); /* s3*(-i) */";
  p "__m256d O1=_mm256_add_pd(s1,_s3ni), O3=_mm256_sub_pd(s1,_s3ni);";
  (* odd combine, IL rule 4: fold the sqrt(1/2) scale into the butterfly via
   * fmadd/fnmadd (MKL's 4-FMA radix-8 shape) — never mul + separate add/sub.
   *   X1 = O1 + O1*(-i); out1 = E1 + C*X1 ; out5 = E1 - C*X1
   *   X3 = O3*(-i) - O3; out3 = E3 + C*X3 ; out7 = E3 - C*X3
   * W2 = O2*(-i) is a paired rotation (rule 2). *)
  p "__m256d _o1ni=_mm256_xor_pd(_mm256_permute_pd(O1,0x5),_M_IM);";
  p "__m256d X1=_mm256_add_pd(O1,_o1ni);";
  p "__m256d W2=_mm256_xor_pd(_mm256_permute_pd(O2,0x5),_M_IM); /* O2*(-i) */";
  p "__m256d _o3ni=_mm256_xor_pd(_mm256_permute_pd(O3,0x5),_M_IM);";
  p "__m256d X3=_mm256_sub_pd(_o3ni,O3);";
  p "out0=_mm256_add_pd(E0,O0); out4=_mm256_sub_pd(E0,O0);";
  p "out1=_mm256_fmadd_pd(_C,X1,E1); out5=_mm256_fnmadd_pd(_C,X1,E1);";
  p "out2=_mm256_add_pd(E2,W2); out6=_mm256_sub_pd(E2,W2);";
  p "out3=_mm256_fmadd_pd(_C,X3,E3); out7=_mm256_fnmadd_pd(_C,X3,E3);"

(* Emit a z-native n1 leaf codelet. ABI mirrors the il-style oop11 shape so it
 * plugs into the existing execution machinery: z in the "re" pointer slot, the
 * "im" slot unused; strides in COMPLEX units (leg Ls, group Gs, out OLs/OGs).
 * Loops `count` columns, 2 per %ymm. *)
(* shared kernel emitter. twiddled=false -> n1 leaf; twiddled=true -> t2
 * (streamed VTW2 twiddles applied to legs 1..R-1 on load, BYTW2 shape).
 * VTW2 stream contract (values plan-time filled, layout fixed here):
 *   per column-pair (k,k+1), per leg l=1..R-1 in leg order, ONE 64-B record:
 *     [c_l(k), c_l(k), c_l(k+1), c_l(k+1)] then [-s_l(k), +s_l(k), -s_l(k+1), +s_l(k+1)]
 *   (cos-first, sign-folded, per-128-bit-lane = per-column) — single forward
 *   cursor, (R-1)*8 doubles per column-pair. Passed via the tw_re slot. *)
let emit_z_kernel ~(vec_width : int) ~(radix : int) ~(twiddled : bool) : string =
  if vec_width <> 4 then failwith "codelet_zil: avx2 only (vec_width 4)";
  if not (List.mem radix [ 4; 8; 16; 32; 64 ]) then
    failwith "codelet_zil: radix must be one of 4/8/16/32/64";
  let kind = if twiddled then "t2" else "n1" in
  let fname = Printf.sprintf "radix%d_z_%s_fwd_avx2" radix kind in
  (* render the body FIRST (staging buffer) so CTw constants discovered during
     lowering can be emitted at file scope before the function *)
  let body = Buffer.create 16384 in
  let consts : (string, string * float * float) Hashtbl.t = Hashtbl.create 16 in
  (if radix = 8 then r8_body body ~ind:"        "
   else emit_dft_body body ~ind:"        " radix consts);
  let buf = Buffer.create 32768 in
  Buffer.add_string buf
    (Printf.sprintf
       "/* Auto-generated by vfft_v2 — TRUE interleaved-native (z) codelet family\n\
       \ * (codelet_zil.ml; docs/roadmap/il_native_design.md §5). 2 complex per\n\
       \ * ymm, no split planes, no boundary conversion. %s, fwd.\n\
       \ * CONTRACT: count %%%% 2 == 0 (2 columns per vector; checklist item 3).%s */\n\
        #include <immintrin.h>\n\
        #include <stddef.h>\n\n\
        static const __m256d _M_IM = { 0.0, -0.0, 0.0, -0.0 }; /* negate im lanes */\n"
       (if twiddled then "t2 (streamed VTW2 twiddles, BYTW2 apply)"
        else "n1 leaf (twiddle-free)")
       (if twiddled then
          "\n * tw_re = the VTW2 stream: per column-pair, per leg l>=1, 64-B\n\
          \ * cos-first sign-folded records in consumption order; tw_im unused."
        else ""));
  Hashtbl.iter
    (fun _key (name, c, s) ->
      Buffer.add_string buf
        (Printf.sprintf
           "static const __m256d %s_c = { %.17g, %.17g, %.17g, %.17g };\n\
            static const __m256d %s_s = { %.17g, %.17g, %.17g, %.17g };\n"
           name c c c c name (-.s) s (-.s) s))
    consts;
  Buffer.add_string buf "\n";
  Buffer.add_string buf
    (Printf.sprintf
       "__attribute__((target(\"avx2,fma\")))\n\
        void %s(\n\
       \    const double * __restrict__ zin,\n\
       \    const double * __restrict__ zin_unused,\n\
       \    double       * __restrict__ zout,\n\
       \    double       * __restrict__ zout_unused,\n\
       \    const double * tw_re, const double * tw_im,\n\
       \    size_t Ls, size_t Gs, size_t OLs, size_t OGs, size_t count)\n\
        {\n\
       \    (void)zin_unused; (void)zout_unused; (void)tw_im; (void)Gs; (void)OGs;%s\n\
       \    for (size_t k = 0; k + 2 <= count; k += 2) {\n"
       fname
       (if twiddled then "" else " (void)tw_re;"));
  if twiddled then
    Buffer.add_string buf
      (Printf.sprintf
         "        const double *twp = tw_re + (k >> 1) * (size_t)%d;\n"
         (8 * (radix - 1)));
  (* loads: point p of columns k,k+1 = zin[2*(p*Ls + k) ..+3]; t2 applies the
     streamed twiddle to legs >= 1 in the load (BYTW2: FLIP + mul + fmadd) *)
  for pt = 0 to radix - 1 do
    if twiddled && pt >= 1 then
      Buffer.add_string buf
        (Printf.sprintf
           "        __m256d x%d = _mm256_loadu_pd(zin + 2*((size_t)%d*Ls + k));\n\
           \        __m256d in%d = _mm256_fmadd_pd(_mm256_loadu_pd(twp + %d), x%d,\n\
           \            _mm256_mul_pd(_mm256_loadu_pd(twp + %d), _mm256_permute_pd(x%d, 0x5)));\n"
           pt pt pt ((pt - 1) * 8) pt (((pt - 1) * 8) + 4) pt)
    else
      Buffer.add_string buf
        (Printf.sprintf
           "        __m256d in%d = _mm256_loadu_pd(zin + 2*((size_t)%d*Ls + k));\n"
           pt pt)
  done;
  for pt = 0 to radix - 1 do
    Buffer.add_string buf (Printf.sprintf "        __m256d out%d;\n" pt)
  done;
  Buffer.add_buffer buf body;
  for pt = 0 to radix - 1 do
    Buffer.add_string buf
      (Printf.sprintf
         "        _mm256_storeu_pd(zout + 2*((size_t)%d*OLs + k), out%d);\n"
         pt pt)
  done;
  Buffer.add_string buf "    }\n}\n";
  Buffer.contents buf

let emit_z_n1 ~(vec_width : int) ~(radix : int) : string =
  emit_z_kernel ~vec_width ~radix ~twiddled:false

let emit_z_t2 ~(vec_width : int) ~(radix : int) : string =
  emit_z_kernel ~vec_width ~radix ~twiddled:true
