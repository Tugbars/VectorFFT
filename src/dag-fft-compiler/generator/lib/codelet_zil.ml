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
 * distinct CTw constants deduped into static VLIT vectors.
 * ~in_of names the input leg i (default "in<i>"); ~park, when given, receives
 * (slot, temp-name) per output instead of the default "out<p> = t;" write —
 * the blocked (Tier-B analog) path uses it to store sub-DFT halves to the
 * function-scope spill arrays immediately after each producer completes. *)
let cid (consts : (string, string * float * float) Hashtbl.t) (c, s) : string =
  let key = Printf.sprintf "%.17g_%.17g" c s in
  match Hashtbl.find_opt consts key with
  | Some (n, _, _) -> n
  | None ->
      let n = Printf.sprintf "_ZW%d" (Hashtbl.length consts) in
      Hashtbl.add consts key (n, c, s);
      n

let emit_dft_body ?(in_of = fun i -> Printf.sprintf "in%d" i) ?park
    (buf : Buffer.t) ~(ind : string) (radix : int)
    (consts : (string, string * float * float) Hashtbl.t) : unit =
  let outs = dft_z radix (Array.init radix (fun i -> In i)) in
  let memo : (zx, string) Hashtbl.t = Hashtbl.create 128 in
  let ctr = ref 0 in
  let fresh () = incr ctr; Printf.sprintf "z%d" !ctr in
  let cid = cid consts in
  let rec go (e : zx) : string =
    match Hashtbl.find_opt memo e with
    | Some v -> v
    | None ->
        let v =
          match e with
          | In i -> in_of i
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
      match park with
      | Some f -> f p v
      | None ->
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
(* BLOCKED (Tier-B analog, doc-58 economics carried to z) body: the DIT
 * recursion's natural pass boundary. PASS 1a/1b compute the even/odd
 * half-DFTs (each own loads + twiddle-apply, own brace scope) and PARK every
 * output to the function-scope zspill[] immediately (z advantage: ONE packed
 * array, not re/im pairs). PASS 2 reloads pairs on demand, applies the
 * top-level combine twiddle (same class selection as dft_z), and stores
 * straight to zout (store-on-compute). *)
let emit_z_blocked_body (buf : Buffer.t) ~(ind : string) ~(radix : int)
    ~(twiddled : bool) (consts : (string, string * float * float) Hashtbl.t) :
    unit =
  let h = radix / 2 in
  let p fmt = Printf.ksprintf (fun s -> Buffer.add_string buf (ind ^ s ^ "\n")) fmt in
  let load_leg name leg =
    if twiddled && leg >= 1 then begin
      p "__m256d x_%s = _mm256_loadu_pd(zin + 2*((size_t)%d*Ls + k));" name leg;
      p "__m256d %s = _mm256_fmadd_pd(_mm256_loadu_pd(twp + %d), x_%s,\n%s    _mm256_mul_pd(_mm256_loadu_pd(twp + %d), _mm256_permute_pd(x_%s, 0x5)));"
        name ((leg - 1) * 8) name ind (((leg - 1) * 8) + 4) name
    end
    else p "__m256d %s = _mm256_loadu_pd(zin + 2*((size_t)%d*Ls + k));" name leg
  in
  let half tag base_slot leg_of =
    p "{ /* PASS 1%s: %s half -> zspill[%d..%d] */" tag
      (if base_slot = 0 then "even" else "odd") base_slot (base_slot + h - 1);
    for i = 0 to h - 1 do
      load_leg (Printf.sprintf "ld%s%d" tag i) (leg_of i)
    done;
    emit_dft_body buf ~ind:(ind ^ "    ") h consts
      ~in_of:(fun i -> Printf.sprintf "ld%s%d" tag i)
      ~park:(fun s v ->
        Buffer.add_string buf
          (Printf.sprintf "%s    _mm256_storeu_pd((double *)&zspill[%d], %s);\n"
             ind (base_slot + s) v));
    p "}"
  in
  half "a" 0 (fun i -> 2 * i);
  half "b" h (fun i -> (2 * i) + 1);
  p "/* PASS 2: combine, reload-on-demand, store-on-compute */";
  let pi = 4.0 *. atan 1.0 in
  for k = 0 to h - 1 do
    p "{";
    p "__m256d ek = _mm256_loadu_pd((const double *)&zspill[%d]);" k;
    p "__m256d ok = _mm256_loadu_pd((const double *)&zspill[%d]);" (h + k);
    let store idx expr =
      p "_mm256_storeu_pd(zout + 2*((size_t)%d*OLs + k), %s);" idx expr
    in
    if k = 0 then begin
      store 0 "_mm256_add_pd(ek,ok)";
      store h "_mm256_sub_pd(ek,ok)"
    end
    else if 4 * k = radix then begin
      p "__m256d t = _mm256_xor_pd(_mm256_permute_pd(ok,0x5),_M_IM);";
      store k "_mm256_add_pd(ek,t)";
      store (k + h) "_mm256_sub_pd(ek,t)"
    end
    else if 8 * k = radix then begin
      p "__m256d x = _mm256_add_pd(ok,_mm256_xor_pd(_mm256_permute_pd(ok,0x5),_M_IM));";
      store k (Printf.sprintf "_mm256_fmadd_pd(_mm256_set1_pd(%.17g),x,ek)" sqh);
      store (k + h) (Printf.sprintf "_mm256_fnmadd_pd(_mm256_set1_pd(%.17g),x,ek)" sqh)
    end
    else if 8 * k = 3 * radix then begin
      p "__m256d x = _mm256_sub_pd(_mm256_xor_pd(_mm256_permute_pd(ok,0x5),_M_IM),ok);";
      store k (Printf.sprintf "_mm256_fmadd_pd(_mm256_set1_pd(%.17g),x,ek)" sqh);
      store (k + h) (Printf.sprintf "_mm256_fnmadd_pd(_mm256_set1_pd(%.17g),x,ek)" sqh)
    end
    else begin
      let c = cos (2.0 *. pi *. float_of_int k /. float_of_int radix)
      and s = -.sin (2.0 *. pi *. float_of_int k /. float_of_int radix) in
      let w = cid consts (c, s) in
      p "__m256d t = _mm256_fmadd_pd(%s_c,ok,_mm256_mul_pd(%s_s,_mm256_permute_pd(ok,0x5)));"
        w w;
      store k "_mm256_add_pd(ek,t)";
      store (k + h) "_mm256_sub_pd(ek,t)"
    end;
    p "}"
  done

(* 2-LEVEL BLOCKED r64 (the 8x8 CT factorization — mono-64's structure at
 * codelet scope). PASS 1: EIGHT spill-free radix-8 sub-DFTs over the residue
 * classes n = 8a+i, each parking output j to zspill[8*i + j]. PASS 2: per
 * output group j, reload the eight slots {8i+j} (the corner-turn is FREE —
 * absorbed into slot indexing, zero lane ops), apply the constant W64^(i*j)
 * twiddles (emit-time CTw class selection), radix-8 combine over i, store
 * X[j + 8m] directly. Both passes are the proven register-clean radix-8 body. *)
let emit_z_blocked2_body (buf : Buffer.t) ~(ind : string) ~(twiddled : bool)
    (consts : (string, string * float * float) Hashtbl.t) : unit =
  let radix = 64 in
  let r = 8 in
  let p fmt = Printf.ksprintf (fun s -> Buffer.add_string buf (ind ^ s ^ "\n")) fmt in
  (* PASS 1: sub-DFT i over legs {8a+i} *)
  for i = 0 to r - 1 do
    p "{ /* PASS 1.%d: legs {8a+%d} -> zspill[%d..%d by 1] */" i i (8 * i)
      ((8 * i) + 7);
    for a = 0 to r - 1 do
      let leg = (8 * a) + i in
      let name = Printf.sprintf "ld%d_%d" i a in
      if twiddled && leg >= 1 then begin
        p "__m256d x_%s = _mm256_loadu_pd(zin + 2*((size_t)%d*Ls + k));" name leg;
        p "__m256d %s = _mm256_fmadd_pd(_mm256_loadu_pd(twp + %d), x_%s,\n%s    _mm256_mul_pd(_mm256_loadu_pd(twp + %d), _mm256_permute_pd(x_%s, 0x5)));"
          name ((leg - 1) * 8) name ind (((leg - 1) * 8) + 4) name
      end
      else
        p "__m256d %s = _mm256_loadu_pd(zin + 2*((size_t)%d*Ls + k));" name leg
    done;
    emit_dft_body buf ~ind:(ind ^ "    ") r consts
      ~in_of:(fun a -> Printf.sprintf "ld%d_%d" i a)
      ~park:(fun j v ->
        Buffer.add_string buf
          (Printf.sprintf "%s    _mm256_storeu_pd((double *)&zspill[%d], %s);\n"
             ind ((8 * i) + j) v));
    p "}"
  done;
  (* PASS 2: group j — reload {8i+j}, constant W64^(i*j) twiddles, radix-8
     over i, store X[j+8m]. *)
  let pi = 4.0 *. atan 1.0 in
  for j = 0 to r - 1 do
    p "{ /* PASS 2.%d: slots {8i+%d} -> X[%d+8m] */" j j j;
    for i = 0 to r - 1 do
      p "__m256d rl%d = _mm256_loadu_pd((const double *)&zspill[%d]);" i
        ((8 * i) + j);
      let e = i * j mod radix in
      if e = 0 then p "__m256d tw%d = rl%d;" i i
      else if 4 * e = radix then
        p "__m256d tw%d = _mm256_xor_pd(_mm256_permute_pd(rl%d,0x5),_M_IM);" i i
      else if 8 * e = radix then begin
        p "__m256d h%d = _mm256_add_pd(rl%d,_mm256_xor_pd(_mm256_permute_pd(rl%d,0x5),_M_IM));"
          i i i;
        p "__m256d tw%d = _mm256_mul_pd(_mm256_set1_pd(%.17g),h%d);" i sqh i
      end
      else if 8 * e = 3 * radix then begin
        p "__m256d h%d = _mm256_sub_pd(_mm256_xor_pd(_mm256_permute_pd(rl%d,0x5),_M_IM),rl%d);"
          i i i;
        p "__m256d tw%d = _mm256_mul_pd(_mm256_set1_pd(%.17g),h%d);" i sqh i
      end
      else begin
        let c = cos (2.0 *. pi *. float_of_int e /. float_of_int radix)
        and s = -.sin (2.0 *. pi *. float_of_int e /. float_of_int radix) in
        let w = cid consts (c, s) in
        p "__m256d tw%d = _mm256_fmadd_pd(%s_c,rl%d,_mm256_mul_pd(%s_s,_mm256_permute_pd(rl%d,0x5)));"
          i w i w i
      end
    done;
    emit_dft_body buf ~ind:(ind ^ "    ") r consts
      ~in_of:(fun i -> Printf.sprintf "tw%d" i)
      ~park:(fun m v ->
        Buffer.add_string buf
          (Printf.sprintf
             "%s    _mm256_storeu_pd(zout + 2*((size_t)%d*OLs + k), %s);\n" ind
             (j + (8 * m)) v));
    p "}"
  done

(* shared kernel emitter. twiddled=false -> n1 leaf; twiddled=true -> t2
 * (streamed VTW2 twiddles applied to legs 1..R-1 on load, BYTW2 shape).
 * VTW2 stream contract (values plan-time filled, layout fixed here):
 *   per column-pair (k,k+1), per leg l=1..R-1 in leg order, ONE 64-B record:
 *     [c_l(k), c_l(k), c_l(k+1), c_l(k+1)] then [-s_l(k), +s_l(k), -s_l(k+1), +s_l(k+1)]
 *   (cos-first, sign-folded, per-128-bit-lane = per-column) — single forward
 *   cursor, (R-1)*8 doubles per column-pair. Passed via the tw_re slot. *)
let emit_z_kernel ~(trans_st : bool) ~(post_tw : bool) ~(strided_st : bool)
    ~(strided : bool) ~(blocked2 : bool) ~(blocked : bool) ~(vec_width : int)
    ~(radix : int) ~(twiddled : bool) : string =
  if vec_width <> 4 then failwith "codelet_zil: avx2 only (vec_width 4)";
  if not (List.mem radix [ 4; 8; 16; 32; 64 ]) then
    failwith "codelet_zil: radix must be one of 4/8/16/32/64";
  if blocked && radix < 16 then
    failwith "codelet_zil: --z-blocked needs radix >= 16 (r4/r8 are spill-free)";
  if blocked2 && radix <> 64 then
    failwith "codelet_zil: --z-blocked2 (8x8 CT) is radix 64 only";
  if (strided || strided_st || post_tw || trans_st) && (blocked || blocked2) then
    failwith "codelet_zil: strided/post-tw/trans-st variants not composed with blocked yet";
  if post_tw && not twiddled then
    failwith "codelet_zil: post_tw (t2d) implies a twiddled kernel";
  if trans_st && (twiddled || strided || strided_st) then
    failwith "codelet_zil: trans_st (n1t corner-turn stores) is an n1 variant";
  let kind =
    (if twiddled then if post_tw then "t2d" else "t2" else "n1")
    ^ (if blocked2 then "b2" else if blocked then "b" else "")
    ^ (if strided then "s" else "")
    ^ (if strided_st then "s" else "")
    ^ if trans_st then "t" else ""
  in
  let fname = Printf.sprintf "radix%d_z_%s_fwd_avx2" radix kind in
  (* render the body FIRST (staging buffer) so CTw constants discovered during
     lowering can be emitted at file scope before the function *)
  let body = Buffer.create 16384 in
  let consts : (string, string * float * float) Hashtbl.t = Hashtbl.create 16 in
  (if blocked2 then emit_z_blocked2_body body ~ind:"        " ~twiddled consts
   else if blocked then emit_z_blocked_body body ~ind:"        " ~radix ~twiddled consts
   else if radix = 8 then r8_body body ~ind:"        "
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
       \    (void)zin_unused; (void)zout_unused; (void)tw_im;%s\n%s\
       \    for (size_t k = 0; k + 2 <= count; k += 2) {\n"
       fname
       ((if strided then "" else " (void)Gs;")
        ^ (if strided_st then "" else " (void)OGs;")
        ^ if twiddled then "" else " (void)tw_re;")
       (if blocked || blocked2 then
          Printf.sprintf "    __m256d zspill[%d]; /* function-scope: L1-hot across iterations */\n" radix
        else ""));
  if twiddled then
    Buffer.add_string buf
      (Printf.sprintf
         "        const double *twp = tw_re + (k >> 1) * (size_t)%d;\n"
         (8 * (radix - 1)));
  if blocked || blocked2 then
    (* blocked bodies carry their own loads (per sub-DFT) and stores (combine) *)
    Buffer.add_buffer buf body
  else begin
    (* loads: point p of columns k,k+1. Contiguous columns: one 256-bit load at
       zin[2*(p*Ls + k)]. STRIDED columns (the "s" variant, FFTW LD shape):
       columns k and k+1 sit Gs complex apart -> two 128-bit loads + insert.
       t2 applies the streamed twiddle to legs >= 1 in the load (BYTW2). *)
    for pt = 0 to radix - 1 do
      let raw = Printf.sprintf "x%d" pt and fin = Printf.sprintf "in%d" pt in
      let pre_tw = twiddled && (not post_tw) && pt >= 1 in
      let dst = if pre_tw then raw else fin in
      (if strided then
         Buffer.add_string buf
           (Printf.sprintf
              "        __m256d %s = _mm256_insertf128_pd(_mm256_castpd128_pd256(\n\
              \            _mm_loadu_pd(zin + 2*((size_t)%d*Ls + k*Gs))),\n\
              \            _mm_loadu_pd(zin + 2*((size_t)%d*Ls + (k+1)*Gs)), 1);\n"
              dst pt pt)
       else
         Buffer.add_string buf
           (Printf.sprintf
              "        __m256d %s = _mm256_loadu_pd(zin + 2*((size_t)%d*Ls + k));\n"
              dst pt));
      if pre_tw then
        Buffer.add_string buf
          (Printf.sprintf
             "        __m256d %s = _mm256_fmadd_pd(_mm256_loadu_pd(twp + %d), %s,\n\
             \            _mm256_mul_pd(_mm256_loadu_pd(twp + %d), _mm256_permute_pd(%s, 0x5)));\n"
             fin ((pt - 1) * 8) raw (((pt - 1) * 8) + 4) raw)
    done;
    for pt = 0 to radix - 1 do
      Buffer.add_string buf (Printf.sprintf "        __m256d out%d;\n" pt)
    done;
    Buffer.add_buffer buf body;
    if trans_st then
      (* CORNER-TURN-IN-STORES (checklist item 4; MKL anatomy §4): process
       * output pairs (p, p+1); two vperm2f128 repack the lanes so BOTH
       * stores are full-width AND contiguous within their column block:
       *   lo = [out_p lane k | out_{p+1} lane k]   -> zout[k*OLs + p]
       *   hi = [out_p lane k+1 | out_{p+1} lane k+1] -> zout[(k+1)*OLs + p]
       * The transposed scratch lets pass 2 run PLAIN t2 (no strided loads). *)
      (for p = 0 to (radix / 2) - 1 do
         let a = 2 * p and b = (2 * p) + 1 in
         Buffer.add_string buf
           (Printf.sprintf
              "        _mm256_storeu_pd(zout + 2*(k*OLs + %d),\n\
              \            _mm256_permute2f128_pd(out%d, out%d, 0x20));\n\
              \        _mm256_storeu_pd(zout + 2*((k+1)*OLs + %d),\n\
              \            _mm256_permute2f128_pd(out%d, out%d, 0x31));\n"
              a a b a a b)
       done)
    else
    for pt = 0 to radix - 1 do
      (* t2d: apply the streamed post-twiddle (BYTW2) to output legs >= 1 *)
      let v =
        if post_tw && pt >= 1 then begin
          Buffer.add_string buf
            (Printf.sprintf
               "        __m256d o%d = _mm256_fmadd_pd(_mm256_loadu_pd(twp + %d), out%d,\n\
               \            _mm256_mul_pd(_mm256_loadu_pd(twp + %d), _mm256_permute_pd(out%d, 0x5)));\n"
               pt ((pt - 1) * 8) pt (((pt - 1) * 8) + 4) pt);
          Printf.sprintf "o%d" pt
        end
        else Printf.sprintf "out%d" pt
      in
      if strided_st then
        (* strided stores (OGs != 1): column pair -> two 128-bit stores *)
        Buffer.add_string buf
          (Printf.sprintf
             "        _mm_storeu_pd(zout + 2*((size_t)%d*OLs + k*OGs), _mm256_castpd256_pd128(%s));\n\
             \        _mm_storeu_pd(zout + 2*((size_t)%d*OLs + (k+1)*OGs), _mm256_extractf128_pd(%s, 1));\n"
             pt v pt v)
      else
        Buffer.add_string buf
          (Printf.sprintf
             "        _mm256_storeu_pd(zout + 2*((size_t)%d*OLs + k), %s);\n" pt v)
    done
  end;
  Buffer.add_string buf "    }\n}\n";
  Buffer.contents buf

let emit_z_n1 ?(strided = false) ?(trans_st = false) ~blocked2 ~blocked
    ~vec_width ~radix () : string =
  emit_z_kernel ~trans_st ~post_tw:false ~strided_st:false ~strided ~blocked2
    ~blocked ~vec_width ~radix ~twiddled:false

let emit_z_t2 ?(strided = false) ?(strided_st = false) ?(post_tw = false)
    ~blocked2 ~blocked ~vec_width ~radix () : string =
  emit_z_kernel ~trans_st:false ~post_tw ~strided_st ~strided ~blocked2
    ~blocked ~vec_width ~radix ~twiddled:true
