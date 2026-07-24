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
    ~(strided : bool) ~(blocked2 : bool) ~(blocked : bool) ~(const_tw : bool)
    ~(pow_tw : bool) ~(pow_tree : bool) ~(tile_ld : bool) ~(vec_width : int)
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
  if const_tw && (not twiddled || post_tw || pow_tw || blocked || blocked2) then
    failwith "codelet_zil: const_tw (t2c) is a plain-t2 variant";
  if pow_tw && (not twiddled || not strided || post_tw || blocked || blocked2) then
    failwith "codelet_zil: pow_tw (t2sp) is a strided-t2 (terminator) variant";
  if pow_tree && (not pow_tw || radix <> 8) then
    failwith "codelet_zil: pow_tree (t2sq) refines pow_tw, radix 8 only";
  if tile_ld && (not strided || radix mod 2 <> 0 || blocked || blocked2) then
    failwith "codelet_zil: tile_ld (t2st/t2spt) is a strided even-radix variant";
  let kind =
    (if twiddled then if post_tw then "t2d" else "t2" else "n1")
    ^ (if const_tw then "c" else "")
    ^ (if blocked2 then "b2" else if blocked then "b" else "")
    ^ (if strided then "s" else "")
    ^ (if strided_st then "s" else "")
    ^ (if pow_tree then "q" else if pow_tw then "p" else "")
    ^ (if tile_ld then "t" else "")
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
       (if twiddled then
          if const_tw then "t2c (GROUP-CONSTANT VTW2 record set, L1-hot, BYTW2 apply)"
          else if pow_tw then
            "t2sp (w^1 VTW2 stream + in-register leg powers, BYTW2 apply)"
          else "t2 (streamed VTW2 twiddles, BYTW2 apply)"
        else "n1 leaf (twiddle-free)")
       (if twiddled then
          if const_tw then
            "\n * tw_re = ONE (R-1)-record VTW2 set for the whole call (group-constant\n\
            \ * twiddles; caller passes its group's set); cursor never advances."
          else if pow_tw then
            "\n * tw_re = ONE 64-B w^1 record per column-pair (cos-first sign-folded);\n\
            \ * legs 2..R-1 built in-register by repeated VTW2 cmul; tw_im unused."
          else
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
    (if const_tw then
       Buffer.add_string buf
         "        const double *twp = tw_re; /* t2c: ONE group-constant record set (L1-hot) */\n"
     else
       Buffer.add_string buf
         (Printf.sprintf
            "        const double *twp = tw_re + (k >> 1) * (size_t)%d;\n"
            (if pow_tw then 8 else 8 * (radix - 1))));
  if pow_tw then begin
    Buffer.add_string buf
      "        __m256d _wc1 = _mm256_loadu_pd(twp);     /* w^1 cos record */\n\
      \        __m256d _ws1 = _mm256_loadu_pd(twp + 4); /* w^1 sin record (sign-folded) */\n";
    if not pow_tree then
      Buffer.add_string buf
        "        __m256d _wc = _wc1, _ws = _ws1;          /* running w^l */\n"
  end;
  if blocked || blocked2 then
    (* blocked bodies carry their own loads (per sub-DFT) and stores (combine) *)
    Buffer.add_buffer buf body
  else begin
    (* loads: point p of columns k,k+1. Contiguous columns: one 256-bit load at
       zin[2*(p*Ls + k)]. STRIDED columns (the "s" variant, FFTW LD shape):
       columns k and k+1 sit Gs complex apart -> two 128-bit loads + insert.
       t2 applies the streamed twiddle to legs >= 1 in the load (BYTW2). *)
    if tile_ld then begin
      (* TILED LOADS (t2st/t2spt; REQUIRES Ls==1, legs contiguous): per
         leg-pair, one WIDE load per column + vperm2f128 repack — R loads +
         R perms per col-pair vs t2s's 2R xmm loads + R inserts. This is the
         MKL-finisher load shape (register corner-turn), mirrored. *)
      for p = 0 to (radix / 2) - 1 do
        let a = 2 * p and b = (2 * p) + 1 in
        Buffer.add_string buf
          (Printf.sprintf
             "        __m256d _ta%d = _mm256_loadu_pd(zin + 2*((size_t)%d*Ls + k*Gs));\n\
             \        __m256d _tb%d = _mm256_loadu_pd(zin + 2*((size_t)%d*Ls + (k+1)*Gs));\n\
             \        __m256d x%d = _mm256_permute2f128_pd(_ta%d, _tb%d, 0x20);\n\
             \        __m256d x%d = _mm256_permute2f128_pd(_ta%d, _tb%d, 0x31);\n"
             p a p a a p p b p p)
      done;
      for pt = 0 to radix - 1 do
        let raw = Printf.sprintf "x%d" pt and fin = Printf.sprintf "in%d" pt in
        let pre_tw = twiddled && (not post_tw) && pt >= 1 in
        if not pre_tw then
          Buffer.add_string buf
            (Printf.sprintf "        __m256d %s = %s;\n" fin raw)
        else if pow_tree then begin
          (* squaring tree: w2=w1*w1, w3=w2*w1, w4=w2*w2, w5=w4*w1, w6=w4*w2,
             w7=w4*w3 — critical path 3 links (vs 6 sequential); the VTW2
             sign-folded form is closed under ARBITRARY products *)
          (if pt >= 2 then
             let a, b = [| 0;0;1;2;2;4;4;4 |].(pt), [| 0;0;1;1;2;1;2;3 |].(pt) in
             Buffer.add_string buf
               (Printf.sprintf
                  "        __m256d _wc%d = _mm256_fnmadd_pd(_ws%d, _ws%d, _mm256_mul_pd(_wc%d, _wc%d));\n\
                  \        __m256d _ws%d = _mm256_fmadd_pd(_wc%d, _ws%d, _mm256_mul_pd(_ws%d, _wc%d));\n"
                  pt a b a b pt a b a b));
          Buffer.add_string buf
            (Printf.sprintf
               "        __m256d %s = _mm256_fmadd_pd(_wc%d, %s,\n\
               \            _mm256_mul_pd(_ws%d, _mm256_permute_pd(%s, 0x5)));\n"
               fin pt raw pt raw)
        end
        else if pow_tw then begin
          if pt >= 2 then
            Buffer.add_string buf
              "        { __m256d _nc = _mm256_fnmadd_pd(_ws, _ws1, _mm256_mul_pd(_wc, _wc1));\n\
              \          _ws = _mm256_fmadd_pd(_wc, _ws1, _mm256_mul_pd(_ws, _wc1));\n\
              \          _wc = _nc; }\n";
          Buffer.add_string buf
            (Printf.sprintf
               "        __m256d %s = _mm256_fmadd_pd(_wc, %s,\n\
               \            _mm256_mul_pd(_ws, _mm256_permute_pd(%s, 0x5)));\n"
               fin raw raw)
        end
        else
          Buffer.add_string buf
            (Printf.sprintf
               "        __m256d %s = _mm256_fmadd_pd(_mm256_loadu_pd(twp + %d), %s,\n\
               \            _mm256_mul_pd(_mm256_loadu_pd(twp + %d), _mm256_permute_pd(%s, 0x5)));\n"
               fin ((pt - 1) * 8) raw (((pt - 1) * 8) + 4) raw)
      done
    end else
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
        if pow_tree then begin
          (* squaring tree (see tile_ld branch): critical path 3 links *)
          (if pt >= 2 then
             let a, b = [| 0;0;1;2;2;4;4;4 |].(pt), [| 0;0;1;1;2;1;2;3 |].(pt) in
             Buffer.add_string buf
               (Printf.sprintf
                  "        __m256d _wc%d = _mm256_fnmadd_pd(_ws%d, _ws%d, _mm256_mul_pd(_wc%d, _wc%d));\n\
                  \        __m256d _ws%d = _mm256_fmadd_pd(_wc%d, _ws%d, _mm256_mul_pd(_ws%d, _wc%d));\n"
                  pt a b a b pt a b a b));
          Buffer.add_string buf
            (Printf.sprintf
               "        __m256d %s = _mm256_fmadd_pd(_wc%d, %s,\n\
               \            _mm256_mul_pd(_ws%d, _mm256_permute_pd(%s, 0x5)));\n"
               fin pt raw pt raw)
        end
        else if pow_tw then begin
          (* t2sp: advance the running twiddle to w^l (VTW2 sign-folded form is
             closed under elementwise cmul: c'=c*c1-s*s1, s'=s*c1+c*s1), then
             BYTW2-apply from registers — 64B/pair streamed instead of (R-1)*64B *)
          if pt >= 2 then
            Buffer.add_string buf
              "        { __m256d _nc = _mm256_fnmadd_pd(_ws, _ws1, _mm256_mul_pd(_wc, _wc1));\n\
              \          _ws = _mm256_fmadd_pd(_wc, _ws1, _mm256_mul_pd(_ws, _wc1));\n\
              \          _wc = _nc; }\n";
          Buffer.add_string buf
            (Printf.sprintf
               "        __m256d %s = _mm256_fmadd_pd(_wc, %s,\n\
               \            _mm256_mul_pd(_ws, _mm256_permute_pd(%s, 0x5)));\n"
               fin raw raw)
        end
        else
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
    ~blocked ~const_tw:false ~pow_tw:false ~pow_tree:false ~tile_ld:false
    ~vec_width ~radix ~twiddled:false

let emit_z_t2 ?(strided = false) ?(strided_st = false) ?(post_tw = false)
    ?(const_tw = false) ?(pow_tw = false) ?(pow_tree = false)
    ?(tile_ld = false) ~blocked2 ~blocked ~vec_width ~radix () : string =
  emit_z_kernel ~trans_st:false ~post_tw ~strided_st ~strided ~blocked2
    ~blocked ~const_tw ~pow_tw ~pow_tree ~tile_ld ~vec_width ~radix
    ~twiddled:true

(* ══════════════════════════════════════════════════════════════════════
   BLOCK-SPLIT INTERIOR family (z_cascade_plan §4.99/§4.996) — PROMOTED from
   the gated hand kernels (zil_split_interior.c spike; paced finals
   zil_split_baked.c). Scratch layout = 64-B [re x4][im x4] blocks (same
   bytes as 4 z-complex; addressing = z's with +4 doubles for the im half;
   ONE stream per leg row — MKL's granularity, measured decisive at 16384).
   Kinds (all standard 11-arg z ABI; kernels use only Ls and count):
     s0s : z-in  -> split-out leaf (twiddle-free; deinterleave in loads,
           shuffles paid ONCE per cascade)
     ms  : split -> split mid, IN-PLACE (caller passes sp as zin AND zout);
           SHUFFLE-FREE: elementwise split cmul, rotations = renames.
           tw_re = per-group SPLAT pairs [c x4][s x4], legs 1..R-1, no cursor.
     msz : split-in -> z-out mid (the cascade's LAST mid; re-interleave in
           the stores). Same twiddle contract as ms.
   Bodies are the hand-derived split-plane forms (fixed per radix, template
   emission — a plane-pair IR backend is not warranted for 2 radices).
   ══════════════════════════════════════════════════════════════════════ *)

let z_split_macros = {|
#define DEINT(zlo, zhi, re, im) do {                                  \
    __m256d _u = _mm256_unpacklo_pd(zlo, zhi);                        \
    __m256d _v = _mm256_unpackhi_pd(zlo, zhi);                        \
    re = _mm256_permute4x64_pd(_u, 0xD8);                             \
    im = _mm256_permute4x64_pd(_v, 0xD8);                             \
} while (0)
#define REINT(re, im, zlo, zhi) do {                                  \
    __m256d _p = _mm256_permute4x64_pd(re, 0xD8);                     \
    __m256d _q = _mm256_permute4x64_pd(im, 0xD8);                     \
    zlo = _mm256_unpacklo_pd(_p, _q);                                 \
    zhi = _mm256_unpackhi_pd(_p, _q);                                 \
} while (0)
#define SPLIT_CMUL(ar,ai, ct,st, or_,oi_) do {                        \
    or_ = _mm256_fnmadd_pd(st, ai, _mm256_mul_pd(ct, ar));            \
    oi_ = _mm256_fmadd_pd(st, ar, _mm256_mul_pd(ct, ai));             \
} while (0)
#define SPLIT_BFLY4(i0r,i0i,i1r,i1i,i2r,i2i,i3r,i3i, o0r,o0i,o1r,o1i,o2r,o2i,o3r,o3i) do { \
    __m256d t0r=_mm256_add_pd(i0r,i2r), t0i=_mm256_add_pd(i0i,i2i);   \
    __m256d t1r=_mm256_sub_pd(i0r,i2r), t1i=_mm256_sub_pd(i0i,i2i);   \
    __m256d t2r=_mm256_add_pd(i1r,i3r), t2i=_mm256_add_pd(i1i,i3i);   \
    __m256d t3r=_mm256_sub_pd(i1r,i3r), t3i=_mm256_sub_pd(i1i,i3i);   \
    o0r=_mm256_add_pd(t0r,t2r); o0i=_mm256_add_pd(t0i,t2i);           \
    o2r=_mm256_sub_pd(t0r,t2r); o2i=_mm256_sub_pd(t0i,t2i);           \
    o1r=_mm256_add_pd(t1r,t3i); o1i=_mm256_sub_pd(t1i,t3r);           \
    o3r=_mm256_sub_pd(t1r,t3i); o3i=_mm256_add_pd(t1i,t3r);           \
} while (0)
#define TR4(a0,a1,a2,a3, t0,t1,t2,t3) do {                            \
    __m256d _u0 = _mm256_unpacklo_pd(a0, a1);                         \
    __m256d _u1 = _mm256_unpackhi_pd(a0, a1);                         \
    __m256d _u2 = _mm256_unpacklo_pd(a2, a3);                         \
    __m256d _u3 = _mm256_unpackhi_pd(a2, a3);                         \
    t0 = _mm256_permute2f128_pd(_u0, _u2, 0x20);                      \
    t1 = _mm256_permute2f128_pd(_u1, _u3, 0x20);                      \
    t2 = _mm256_permute2f128_pd(_u0, _u2, 0x31);                      \
    t3 = _mm256_permute2f128_pd(_u1, _u3, 0x31);                      \
} while (0)
#define WPROD(cA,sA, cB,sB, cP,sP) do {                               \
    cP = _mm256_fnmadd_pd(sA, sB, _mm256_mul_pd(cA, cB));             \
    sP = _mm256_fmadd_pd(cA, sB, _mm256_mul_pd(sA, cB));              \
} while (0)
#define SPLIT_BFLY8(x0r,x0i,x1r,x1i,x2r,x2i,x3r,x3i,x4r,x4i,x5r,x5i,x6r,x6i,x7r,x7i, \
                    o0r,o0i,o1r,o1i,o2r,o2i,o3r,o3i,o4r,o4i,o5r,o5i,o6r,o6i,o7r,o7i) do { \
    const __m256d _C = _mm256_set1_pd(0.70710678118654752440);        \
    __m256d t0r=_mm256_add_pd(x0r,x4r), t0i=_mm256_add_pd(x0i,x4i);   \
    __m256d t1r=_mm256_sub_pd(x0r,x4r), t1i=_mm256_sub_pd(x0i,x4i);   \
    __m256d t2r=_mm256_add_pd(x2r,x6r), t2i=_mm256_add_pd(x2i,x6i);   \
    __m256d t3r=_mm256_sub_pd(x2r,x6r), t3i=_mm256_sub_pd(x2i,x6i);   \
    __m256d E0r=_mm256_add_pd(t0r,t2r), E0i=_mm256_add_pd(t0i,t2i);   \
    __m256d E2r=_mm256_sub_pd(t0r,t2r), E2i=_mm256_sub_pd(t0i,t2i);   \
    __m256d E1r=_mm256_add_pd(t1r,t3i), E1i=_mm256_sub_pd(t1i,t3r);   \
    __m256d E3r=_mm256_sub_pd(t1r,t3i), E3i=_mm256_add_pd(t1i,t3r);   \
    __m256d s0r=_mm256_add_pd(x1r,x5r), s0i=_mm256_add_pd(x1i,x5i);   \
    __m256d s1r=_mm256_sub_pd(x1r,x5r), s1i=_mm256_sub_pd(x1i,x5i);   \
    __m256d s2r=_mm256_add_pd(x3r,x7r), s2i=_mm256_add_pd(x3i,x7i);   \
    __m256d s3r=_mm256_sub_pd(x3r,x7r), s3i=_mm256_sub_pd(x3i,x7i);   \
    __m256d O0r=_mm256_add_pd(s0r,s2r), O0i=_mm256_add_pd(s0i,s2i);   \
    __m256d O2r=_mm256_sub_pd(s0r,s2r), O2i=_mm256_sub_pd(s0i,s2i);   \
    __m256d O1r=_mm256_add_pd(s1r,s3i), O1i=_mm256_sub_pd(s1i,s3r);   \
    __m256d O3r=_mm256_sub_pd(s1r,s3i), O3i=_mm256_add_pd(s1i,s3r);   \
    __m256d X1r=_mm256_add_pd(O1r,O1i), X1i=_mm256_sub_pd(O1i,O1r);   \
    __m256d X3r=_mm256_sub_pd(O3i,O3r), X3n=_mm256_add_pd(O3r,O3i);   \
    o0r=_mm256_add_pd(E0r,O0r); o0i=_mm256_add_pd(E0i,O0i);           \
    o4r=_mm256_sub_pd(E0r,O0r); o4i=_mm256_sub_pd(E0i,O0i);           \
    o1r=_mm256_fmadd_pd(_C,X1r,E1r); o1i=_mm256_fmadd_pd(_C,X1i,E1i); \
    o5r=_mm256_fnmadd_pd(_C,X1r,E1r); o5i=_mm256_fnmadd_pd(_C,X1i,E1i); \
    o2r=_mm256_add_pd(E2r,O2i); o2i=_mm256_sub_pd(E2i,O2r);           \
    o6r=_mm256_sub_pd(E2r,O2i); o6i=_mm256_add_pd(E2i,O2r);           \
    o3r=_mm256_fmadd_pd(_C,X3r,E3r); o3i=_mm256_fnmadd_pd(_C,X3n,E3i); \
    o7r=_mm256_fnmadd_pd(_C,X3r,E3r); o7i=_mm256_fmadd_pd(_C,X3n,E3i); \
} while (0)
/* INVERSE butterflies (IDFT bodies): derived from the fwd macros by the
 * conj rule IDFT = conj(DFT(conj(.))) — every CROSS-PLANE term flips sign,
 * same-plane terms unchanged. Same op counts, zero shuffles. */
#define SPLIT_BFLY4_INV(i0r,i0i,i1r,i1i,i2r,i2i,i3r,i3i, o0r,o0i,o1r,o1i,o2r,o2i,o3r,o3i) do { \
    __m256d t0r=_mm256_add_pd(i0r,i2r), t0i=_mm256_add_pd(i0i,i2i);   \
    __m256d t1r=_mm256_sub_pd(i0r,i2r), t1i=_mm256_sub_pd(i0i,i2i);   \
    __m256d t2r=_mm256_add_pd(i1r,i3r), t2i=_mm256_add_pd(i1i,i3i);   \
    __m256d t3r=_mm256_sub_pd(i1r,i3r), t3i=_mm256_sub_pd(i1i,i3i);   \
    o0r=_mm256_add_pd(t0r,t2r); o0i=_mm256_add_pd(t0i,t2i);           \
    o2r=_mm256_sub_pd(t0r,t2r); o2i=_mm256_sub_pd(t0i,t2i);           \
    o1r=_mm256_sub_pd(t1r,t3i); o1i=_mm256_add_pd(t1i,t3r);  /* +(+i)t3 */ \
    o3r=_mm256_add_pd(t1r,t3i); o3i=_mm256_sub_pd(t1i,t3r);  /* -(+i)t3 */ \
} while (0)
#define SPLIT_BFLY8_INV(x0r,x0i,x1r,x1i,x2r,x2i,x3r,x3i,x4r,x4i,x5r,x5i,x6r,x6i,x7r,x7i, \
                    o0r,o0i,o1r,o1i,o2r,o2i,o3r,o3i,o4r,o4i,o5r,o5i,o6r,o6i,o7r,o7i) do { \
    const __m256d _C = _mm256_set1_pd(0.70710678118654752440);        \
    __m256d t0r=_mm256_add_pd(x0r,x4r), t0i=_mm256_add_pd(x0i,x4i);   \
    __m256d t1r=_mm256_sub_pd(x0r,x4r), t1i=_mm256_sub_pd(x0i,x4i);   \
    __m256d t2r=_mm256_add_pd(x2r,x6r), t2i=_mm256_add_pd(x2i,x6i);   \
    __m256d t3r=_mm256_sub_pd(x2r,x6r), t3i=_mm256_sub_pd(x2i,x6i);   \
    __m256d E0r=_mm256_add_pd(t0r,t2r), E0i=_mm256_add_pd(t0i,t2i);   \
    __m256d E2r=_mm256_sub_pd(t0r,t2r), E2i=_mm256_sub_pd(t0i,t2i);   \
    __m256d E1r=_mm256_sub_pd(t1r,t3i), E1i=_mm256_add_pd(t1i,t3r);   \
    __m256d E3r=_mm256_add_pd(t1r,t3i), E3i=_mm256_sub_pd(t1i,t3r);   \
    __m256d s0r=_mm256_add_pd(x1r,x5r), s0i=_mm256_add_pd(x1i,x5i);   \
    __m256d s1r=_mm256_sub_pd(x1r,x5r), s1i=_mm256_sub_pd(x1i,x5i);   \
    __m256d s2r=_mm256_add_pd(x3r,x7r), s2i=_mm256_add_pd(x3i,x7i);   \
    __m256d s3r=_mm256_sub_pd(x3r,x7r), s3i=_mm256_sub_pd(x3i,x7i);   \
    __m256d O0r=_mm256_add_pd(s0r,s2r), O0i=_mm256_add_pd(s0i,s2i);   \
    __m256d O2r=_mm256_sub_pd(s0r,s2r), O2i=_mm256_sub_pd(s0i,s2i);   \
    __m256d O1r=_mm256_sub_pd(s1r,s3i), O1i=_mm256_add_pd(s1i,s3r);   \
    __m256d O3r=_mm256_add_pd(s1r,s3i), O3i=_mm256_sub_pd(s1i,s3r);   \
    __m256d X1r=_mm256_sub_pd(O1r,O1i), X1i=_mm256_add_pd(O1i,O1r);   \
    __m256d X3r=_mm256_add_pd(O3i,O3r), X3n=_mm256_sub_pd(O3r,O3i);   \
    o0r=_mm256_add_pd(E0r,O0r); o0i=_mm256_add_pd(E0i,O0i);           \
    o4r=_mm256_sub_pd(E0r,O0r); o4i=_mm256_sub_pd(E0i,O0i);           \
    o1r=_mm256_fmadd_pd(_C,X1r,E1r); o1i=_mm256_fmadd_pd(_C,X1i,E1i); \
    o5r=_mm256_fnmadd_pd(_C,X1r,E1r); o5i=_mm256_fnmadd_pd(_C,X1i,E1i); \
    o2r=_mm256_sub_pd(E2r,O2i); o2i=_mm256_add_pd(E2i,O2r);           \
    o6r=_mm256_add_pd(E2r,O2i); o6i=_mm256_sub_pd(E2i,O2r);           \
    o3r=_mm256_fnmadd_pd(_C,X3r,E3r); o3i=_mm256_fmadd_pd(_C,X3n,E3i); \
    o7r=_mm256_fmadd_pd(_C,X3r,E3r); o7i=_mm256_fnmadd_pd(_C,X3n,E3i); \
} while (0)
|}

let z_split_bfly_call ?(inv = false) radix =
  let m4 = if inv then "SPLIT_BFLY4_INV" else "SPLIT_BFLY4" in
  let m8 = if inv then "SPLIT_BFLY8_INV" else "SPLIT_BFLY8" in
  if radix = 4 then
    Printf.sprintf
      "        %s(xr[0],xi[0],xr[1],xi[1],xr[2],xi[2],xr[3],xi[3],\n\
      \                    or_[0],oi_[0],or_[1],oi_[1],or_[2],oi_[2],or_[3],oi_[3]);\n"
      m4
  else
    Printf.sprintf
      "        %s(xr[0],xi[0],xr[1],xi[1],xr[2],xi[2],xr[3],xi[3],\n\
      \                    xr[4],xi[4],xr[5],xi[5],xr[6],xi[6],xr[7],xi[7],\n\
      \                    or_[0],oi_[0],or_[1],oi_[1],or_[2],oi_[2],or_[3],oi_[3],\n\
      \                    or_[4],oi_[4],or_[5],oi_[5],or_[6],oi_[6],or_[7],oi_[7]);\n"
      m8

let emit_z_split ~(kind : string) ~(radix : int) () : string =
  if not (List.mem radix [ 4; 8 ]) then
    failwith "codelet_zil: split family covers radix 4/8 (r16 split = 32 live planes, spills)";
  if not (List.mem kind [ "s0s"; "ms"; "msz"; "sterm"; "s0sb"; "msb"; "stermb" ])
  then failwith "codelet_zil: split kind must be s0s|ms|msz|sterm|s0sb|msb|stermb";
  if (kind = "sterm" || kind = "stermb") && radix <> 8 then
    failwith "codelet_zil: sterm/stermb (split terminator pair) is radix 8 only";
  let r = radix in
  let bwd = List.mem kind [ "s0sb"; "msb"; "stermb" ] in
  let base_kind =
    if bwd then String.sub kind 0 (String.length kind - 1) else kind
  in
  let fname =
    Printf.sprintf "radix%d_z_%s_%s_avx2" r base_kind
      (if bwd then "bwd" else "fwd")
  in
  if kind = "stermb" then
    (* INVERSE split terminator (the bwd cascade's FIRST stage): consumes the
       digit-reversed z comb, inverse radix-8 DFT, POST-multiplies legs 1..7
       by the CONJUGATED packed per-column w^1 (conjugation is TABLE-SIDE:
       the plan fills +sin), transposes leg-vectors back to blocks, stores
       the block-split plane. Comb loads are contiguous per leg (cheaper
       than fwd's load transposes). ABI: zin = z comb, zout = split plane,
       tw_re = packed conj table at tw_re+2k, OLs = N/8, count = N/8. *)
    Printf.sprintf
      "/* Auto-generated by vfft_v2 — BLOCK-SPLIT interior family (codelet_zil.ml).\n\
      \ * stermb: INVERSE split terminator (z drev comb -> split plane), IDFT bodies,\n\
      \ * post-multiplied CONJ packed per-column w^1 (table-side conj), transpose in\n\
      \ * stores. CONTRACT: count %%%% 4 == 0. bwd(fwd) = N*x. */\n\
       #include <immintrin.h>\n\
       #include <stddef.h>\n%s\n\
       __attribute__((target(\"avx2,fma\")))\n\
       void %s(\n\
      \    const double * __restrict__ zin,\n\
      \    const double * __restrict__ zin_unused,\n\
      \    double       * __restrict__ zout,\n\
      \    double       * __restrict__ zout_unused,\n\
      \    const double * tw_re, const double * tw_im,\n\
      \    size_t Ls, size_t Gs, size_t OLs, size_t OGs, size_t count)\n\
       {\n\
      \    (void)zin_unused; (void)zout_unused; (void)tw_im; (void)Ls; (void)Gs; (void)OGs;\n\
      \    for (size_t k = 0; k + 4 <= count; k += 4) {\n\
      \        __m256d xr[8], xi[8];\n\
      \        for (int l = 0; l < 8; l++) {\n\
      \            __m256d zlo = _mm256_loadu_pd(zin + 2*((size_t)l*OLs + k));\n\
      \            __m256d zhi = _mm256_loadu_pd(zin + 2*((size_t)l*OLs + k) + 4);\n\
      \            DEINT(zlo, zhi, xr[l], xi[l]);\n\
      \        }\n\
      \        __m256d or_[8], oi_[8];\n%s\
      \        {\n\
      \            __m256d c1 = _mm256_loadu_pd(tw_re + 2*(size_t)k);\n\
      \            __m256d s1 = _mm256_loadu_pd(tw_re + 2*(size_t)k + 4);\n\
      \            __m256d c2, s2, c3, s3, c4, s4, cw, sw, rr, ii;\n\
      \            SPLIT_CMUL(or_[1], oi_[1], c1, s1, rr, ii); or_[1] = rr; oi_[1] = ii;\n\
      \            WPROD(c1, s1, c1, s1, c2, s2);\n\
      \            SPLIT_CMUL(or_[2], oi_[2], c2, s2, rr, ii); or_[2] = rr; oi_[2] = ii;\n\
      \            WPROD(c2, s2, c1, s1, c3, s3);\n\
      \            SPLIT_CMUL(or_[3], oi_[3], c3, s3, rr, ii); or_[3] = rr; oi_[3] = ii;\n\
      \            WPROD(c2, s2, c2, s2, c4, s4);\n\
      \            SPLIT_CMUL(or_[4], oi_[4], c4, s4, rr, ii); or_[4] = rr; oi_[4] = ii;\n\
      \            WPROD(c4, s4, c1, s1, cw, sw);\n\
      \            SPLIT_CMUL(or_[5], oi_[5], cw, sw, rr, ii); or_[5] = rr; oi_[5] = ii;\n\
      \            WPROD(c4, s4, c2, s2, cw, sw);\n\
      \            SPLIT_CMUL(or_[6], oi_[6], cw, sw, rr, ii); or_[6] = rr; oi_[6] = ii;\n\
      \            WPROD(c4, s4, c3, s3, cw, sw);\n\
      \            SPLIT_CMUL(or_[7], oi_[7], cw, sw, rr, ii); or_[7] = rr; oi_[7] = ii;\n\
      \        }\n\
      \        {\n\
      \            __m256d b0, b1, b2, b3;\n\
      \            TR4(or_[0], or_[1], or_[2], or_[3], b0, b1, b2, b3);\n\
      \            _mm256_storeu_pd(zout + 16*(size_t)k,        b0);\n\
      \            _mm256_storeu_pd(zout + 16*((size_t)k + 1),  b1);\n\
      \            _mm256_storeu_pd(zout + 16*((size_t)k + 2),  b2);\n\
      \            _mm256_storeu_pd(zout + 16*((size_t)k + 3),  b3);\n\
      \            TR4(oi_[0], oi_[1], oi_[2], oi_[3], b0, b1, b2, b3);\n\
      \            _mm256_storeu_pd(zout + 16*(size_t)k + 4,       b0);\n\
      \            _mm256_storeu_pd(zout + 16*((size_t)k + 1) + 4, b1);\n\
      \            _mm256_storeu_pd(zout + 16*((size_t)k + 2) + 4, b2);\n\
      \            _mm256_storeu_pd(zout + 16*((size_t)k + 3) + 4, b3);\n\
      \            TR4(or_[4], or_[5], or_[6], or_[7], b0, b1, b2, b3);\n\
      \            _mm256_storeu_pd(zout + 16*(size_t)k + 8,       b0);\n\
      \            _mm256_storeu_pd(zout + 16*((size_t)k + 1) + 8, b1);\n\
      \            _mm256_storeu_pd(zout + 16*((size_t)k + 2) + 8, b2);\n\
      \            _mm256_storeu_pd(zout + 16*((size_t)k + 3) + 8, b3);\n\
      \            TR4(oi_[4], oi_[5], oi_[6], oi_[7], b0, b1, b2, b3);\n\
      \            _mm256_storeu_pd(zout + 16*(size_t)k + 12,       b0);\n\
      \            _mm256_storeu_pd(zout + 16*((size_t)k + 1) + 12, b1);\n\
      \            _mm256_storeu_pd(zout + 16*((size_t)k + 2) + 12, b2);\n\
      \            _mm256_storeu_pd(zout + 16*((size_t)k + 3) + 12, b3);\n\
      \        }\n\
      \    }\n\
       }\n"
      z_split_macros fname (z_split_bfly_call ~inv:true 8)
  else if kind = "sterm" then
    (* split-input TERMINATOR (measured winner, terminator race 2026-07-24:
       +4-7% over t2sp/t2spt at every cell): reads the block-split plane
       directly (ALL mids stay plain ms — no msz pass), 4 columns/iteration,
       4x4 register transposes on load, shuffle-free split butterfly +
       twiddles, PACKED per-column w^1 table (16 B/col, half of w^1-VTW2),
       squaring-tree powers, re-interleave fused in the stores.
       ABI mapping: zin = block-split plane, zout = z out (digit-reversed
       comb), tw_re = packed table ([c x4][s x4] per 4 cols at tw_re + 2k),
       OLs = N/R, count = N/R. Ls/Gs/OGs unused. *)
    Printf.sprintf
      "/* Auto-generated by vfft_v2 — BLOCK-SPLIT interior family (codelet_zil.ml;\n\
      \ * z_cascade_plan §4.99/§4.998). sterm: SPLIT-INPUT terminator, 4 cols/iter,\n\
      \ * packed per-column w^1 twiddles (16 B/col), shuffle-free split math,\n\
      \ * re-interleave fused in stores. CONTRACT: count %%%% 4 == 0; mids = ms only.\n\
      \ * tw_re layout: per 4 columns at tw_re + 2k: [c(k..k+3)][s(k..k+3)]. */\n\
       #include <immintrin.h>\n\
       #include <stddef.h>\n%s\n\
       __attribute__((target(\"avx2,fma\")))\n\
       void %s(\n\
      \    const double * __restrict__ zin,\n\
      \    const double * __restrict__ zin_unused,\n\
      \    double       * __restrict__ zout,\n\
      \    double       * __restrict__ zout_unused,\n\
      \    const double * tw_re, const double * tw_im,\n\
      \    size_t Ls, size_t Gs, size_t OLs, size_t OGs, size_t count)\n\
       {\n\
      \    (void)zin_unused; (void)zout_unused; (void)tw_im; (void)Ls; (void)Gs; (void)OGs;\n\
      \    for (size_t k = 0; k + 4 <= count; k += 4) {\n\
      \        __m256d xr[8], xi[8];\n\
      \        {\n\
      \            __m256d rl0 = _mm256_loadu_pd(zin + 16*(size_t)k);\n\
      \            __m256d il0 = _mm256_loadu_pd(zin + 16*(size_t)k + 4);\n\
      \            __m256d rh0 = _mm256_loadu_pd(zin + 16*(size_t)k + 8);\n\
      \            __m256d ih0 = _mm256_loadu_pd(zin + 16*(size_t)k + 12);\n\
      \            __m256d rl1 = _mm256_loadu_pd(zin + 16*((size_t)k+1));\n\
      \            __m256d il1 = _mm256_loadu_pd(zin + 16*((size_t)k+1) + 4);\n\
      \            __m256d rh1 = _mm256_loadu_pd(zin + 16*((size_t)k+1) + 8);\n\
      \            __m256d ih1 = _mm256_loadu_pd(zin + 16*((size_t)k+1) + 12);\n\
      \            __m256d rl2 = _mm256_loadu_pd(zin + 16*((size_t)k+2));\n\
      \            __m256d il2 = _mm256_loadu_pd(zin + 16*((size_t)k+2) + 4);\n\
      \            __m256d rh2 = _mm256_loadu_pd(zin + 16*((size_t)k+2) + 8);\n\
      \            __m256d ih2 = _mm256_loadu_pd(zin + 16*((size_t)k+2) + 12);\n\
      \            __m256d rl3 = _mm256_loadu_pd(zin + 16*((size_t)k+3));\n\
      \            __m256d il3 = _mm256_loadu_pd(zin + 16*((size_t)k+3) + 4);\n\
      \            __m256d rh3 = _mm256_loadu_pd(zin + 16*((size_t)k+3) + 8);\n\
      \            __m256d ih3 = _mm256_loadu_pd(zin + 16*((size_t)k+3) + 12);\n\
      \            TR4(rl0, rl1, rl2, rl3, xr[0], xr[1], xr[2], xr[3]);\n\
      \            TR4(il0, il1, il2, il3, xi[0], xi[1], xi[2], xi[3]);\n\
      \            TR4(rh0, rh1, rh2, rh3, xr[4], xr[5], xr[6], xr[7]);\n\
      \            TR4(ih0, ih1, ih2, ih3, xi[4], xi[5], xi[6], xi[7]);\n\
      \        }\n\
      \        {\n\
      \            __m256d c1 = _mm256_loadu_pd(tw_re + 2*(size_t)k);\n\
      \            __m256d s1 = _mm256_loadu_pd(tw_re + 2*(size_t)k + 4);\n\
      \            __m256d c2, s2, c3, s3, c4, s4, cw, sw, rr, ii;\n\
      \            SPLIT_CMUL(xr[1], xi[1], c1, s1, rr, ii); xr[1] = rr; xi[1] = ii;\n\
      \            WPROD(c1, s1, c1, s1, c2, s2);\n\
      \            SPLIT_CMUL(xr[2], xi[2], c2, s2, rr, ii); xr[2] = rr; xi[2] = ii;\n\
      \            WPROD(c2, s2, c1, s1, c3, s3);\n\
      \            SPLIT_CMUL(xr[3], xi[3], c3, s3, rr, ii); xr[3] = rr; xi[3] = ii;\n\
      \            WPROD(c2, s2, c2, s2, c4, s4);\n\
      \            SPLIT_CMUL(xr[4], xi[4], c4, s4, rr, ii); xr[4] = rr; xi[4] = ii;\n\
      \            WPROD(c4, s4, c1, s1, cw, sw);\n\
      \            SPLIT_CMUL(xr[5], xi[5], cw, sw, rr, ii); xr[5] = rr; xi[5] = ii;\n\
      \            WPROD(c4, s4, c2, s2, cw, sw);\n\
      \            SPLIT_CMUL(xr[6], xi[6], cw, sw, rr, ii); xr[6] = rr; xi[6] = ii;\n\
      \            WPROD(c4, s4, c3, s3, cw, sw);\n\
      \            SPLIT_CMUL(xr[7], xi[7], cw, sw, rr, ii); xr[7] = rr; xi[7] = ii;\n\
      \        }\n\
      \        __m256d or_[8], oi_[8];\n%s\
      \        for (int l = 0; l < 8; l++) {\n\
      \            __m256d zlo, zhi;\n\
      \            REINT(or_[l], oi_[l], zlo, zhi);\n\
      \            _mm256_storeu_pd(zout + 2*((size_t)l*OLs + k), zlo);\n\
      \            _mm256_storeu_pd(zout + 2*((size_t)l*OLs + k) + 4, zhi);\n\
      \        }\n\
      \    }\n\
       }\n"
      z_split_macros fname (z_split_bfly_call 8)
  else
  let buf = Buffer.create 16384 in
  Buffer.add_string buf
    (Printf.sprintf
       "/* Auto-generated by vfft_v2 — BLOCK-SPLIT interior family (codelet_zil.ml;\n\
       \ * z_cascade_plan §4.99/§4.996). Scratch = 64-B [re x4][im x4] blocks (z\n\
       \ * addressing +4 for im; one stream per leg row). %s, fwd.\n\
       \ * CONTRACT: count %%%% 4 == 0 (4 columns per iteration).%s */\n\
        #include <immintrin.h>\n\
        #include <stddef.h>\n%s\n"
       (match kind with
        | "s0s" -> "s0s (z-in -> split-out leaf, twiddle-free, deinterleaving loads)"
        | "ms" -> "ms (split mid, IN-PLACE zin==zout, SHUFFLE-FREE, splat-pair tw)"
        | "msb" ->
          "msb (INVERSE split mid: IDFT body, POST-multiplied CONJ splat-pair tw)"
        | "s0sb" ->
          "s0sb (INVERSE leaf: split-in -> z-out, IDFT body, twiddle-free)"
        | _ -> "msz (split-in -> z-out LAST mid, re-interleaving stores, splat-pair tw)")
       (if kind = "s0s" || kind = "s0sb" then ""
        else
          "\n * tw_re = ONE per-group splat-pair set: legs 1..R-1, 8 doubles/leg\n\
          \ * [c,c,c,c][s,s,s,s]; no cursor (group-constant; bwd = CONJ values,\n\
          \ * table-side). tw_im unused.")
       z_split_macros);
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
       \    (void)zin_unused; (void)zout_unused; (void)tw_im; (void)Gs; (void)OLs; (void)OGs;%s\n\
       \    for (size_t k = 0; k + 4 <= count; k += 4) {\n\
       \        __m256d xr[%d], xi[%d], or_[%d], oi_[%d];\n"
       fname
       (if kind = "s0s" || kind = "s0sb" then " (void)tw_re;" else "")
       r r r r);
  (* loads *)
  (match kind with
   | "s0s" ->
     Buffer.add_string buf
       (Printf.sprintf
          "        for (int l = 0; l < %d; l++) {\n\
          \            __m256d zlo = _mm256_loadu_pd(zin + 2*((size_t)l*Ls + k));\n\
          \            __m256d zhi = _mm256_loadu_pd(zin + 2*((size_t)l*Ls + k) + 4);\n\
          \            DEINT(zlo, zhi, xr[l], xi[l]);\n\
          \        }\n" r)
   | "s0sb" | "msb" ->
     (* bwd: plain split loads, NO twiddle on load (post-applied for msb) *)
     Buffer.add_string buf
       (Printf.sprintf
          "        for (int l = 0; l < %d; l++) {\n\
          \            xr[l] = _mm256_loadu_pd(zin + 2*((size_t)l*Ls + k));\n\
          \            xi[l] = _mm256_loadu_pd(zin + 2*((size_t)l*Ls + k) + 4);\n\
          \        }\n" r)
   | _ ->
     Buffer.add_string buf
       (Printf.sprintf
          "        xr[0] = _mm256_loadu_pd(zin + 2*(size_t)k);\n\
          \        xi[0] = _mm256_loadu_pd(zin + 2*(size_t)k + 4);\n\
          \        for (int l = 1; l < %d; l++) {\n\
          \            __m256d ar = _mm256_loadu_pd(zin + 2*((size_t)l*Ls + k));\n\
          \            __m256d ai = _mm256_loadu_pd(zin + 2*((size_t)l*Ls + k) + 4);\n\
          \            __m256d ct = _mm256_loadu_pd(tw_re + (size_t)(l - 1) * 8);\n\
          \            __m256d st = _mm256_loadu_pd(tw_re + (size_t)(l - 1) * 8 + 4);\n\
          \            SPLIT_CMUL(ar, ai, ct, st, xr[l], xi[l]);\n\
          \        }\n" r));
  Buffer.add_string buf (z_split_bfly_call ~inv:bwd r);
  (* msb: POST-multiply output legs 1..R-1 by the (conj, table-side) set *)
  if kind = "msb" then
    Buffer.add_string buf
      (Printf.sprintf
         "        for (int l = 1; l < %d; l++) {\n\
         \            __m256d ct = _mm256_loadu_pd(tw_re + (size_t)(l - 1) * 8);\n\
         \            __m256d st = _mm256_loadu_pd(tw_re + (size_t)(l - 1) * 8 + 4);\n\
         \            __m256d rr, ii;\n\
         \            SPLIT_CMUL(or_[l], oi_[l], ct, st, rr, ii);\n\
         \            or_[l] = rr; oi_[l] = ii;\n\
         \        }\n" r);
  (* stores *)
  (match kind with
   | "msz" | "s0sb" ->
     Buffer.add_string buf
       (Printf.sprintf
          "        for (int l = 0; l < %d; l++) {\n\
          \            __m256d zlo, zhi;\n\
          \            REINT(or_[l], oi_[l], zlo, zhi);\n\
          \            _mm256_storeu_pd(zout + 2*((size_t)l*Ls + k), zlo);\n\
          \            _mm256_storeu_pd(zout + 2*((size_t)l*Ls + k) + 4, zhi);\n\
          \        }\n" r)
   | _ ->
     Buffer.add_string buf
       (Printf.sprintf
          "        for (int l = 0; l < %d; l++) {\n\
          \            _mm256_storeu_pd(zout + 2*((size_t)l*Ls + k), or_[l]);\n\
          \            _mm256_storeu_pd(zout + 2*((size_t)l*Ls + k) + 4, oi_[l]);\n\
          \        }\n" r));
  Buffer.add_string buf "    }\n}\n";
  Buffer.contents buf
