(* cx_ir.ml — the packed-complex IR of the full-IL (cil) family.
 *
 * Split out of codelet_cil.ml (Phase 0 decomposition, 2026-08-09,
 * byte-identity gated). One node type, hash-consed: structural equality
 * becomes tag equality, which is CSE (mirrors Ir.hashcons). Also hosts the
 * emission-STATE refs (tw_log3 / tw_pre / st_turn / st_turn_gs) — they are
 * render/store-form state, not IR state, but they live beside the IR so
 * every cx_* module sees one copy.
 * MODULE CARD
 * ROLE: cx_kind + t + hash-cons (mk/reset) + smart constructors + state refs.
 * GOTCHA 1: `reset ()` MUST run before each codelet/pass — table and tag
 * counter are module-global, exactly like Ir.reset (tags name the zN
 * C locals, so a stale table leaks names across brace scopes). *)

(* ═══════════════════════════════════════════════════════════════
 *  THE COMPLEX IR
 * ═══════════════════════════════════════════════════════════════ *)

(* ── Symbolic addresses — the memory world, as DATA ─────────────────────
 * One constructor per address FORM the cil family emits. The runtime names
 * (k, Ls, OLs, OGs, twp) are fixed by the frozen z ABI, so a form + its
 * compile-time ints IS the address; rendering to a C string is cx_render's
 * job. `col` on the turned forms selects column k (0) or k+1 (1) — the
 * corner-turn writes two columns per iteration. *)
type caddr =
  | AZinLeg of int (* zin [2*((size_t)l*Ls  + k)]                  *)
  | AZoutLeg of int (* zout[2*((size_t)l*OLs + k)]                  *)
  | AZoutTurn of int * int (* (l, col)  zout[2*(((size_t)k+c)*OLs + l)]     *)
  | AZoutTurnG of int * int (* (l, col)  ... + (size_t)l*OGs)]  t2tg scatter  *)
  | AS of int (* S[i]  — the blocked spill plane, flat doubles *)
  | AP of int (* P[i]  — emit_k1's stage plane                 *)
  | AZinAbs of int (* zin [i] — emit_k1 absolute (no k)             *)
  | AZoutAbs of int (* zout[i] — emit_k1 absolute                    *)
  | ATw of int (* twp [i] — the T2 streamed VTW2 cursor          *)
  (* ── the real pair's top stage t2h (real_il.ml, 2026-09-29) ──
     MIRROR forms: the value is reversed across the vector's complex lanes
     and conjugated on the way in (load) or out (store); the address is the
     mirror of column k's group, m*pitch - k - (per-1) with per = complex
     per vector (at VEX-128 the reversal is the identity). The SPECIAL
     forms serve the self-mirrored columns 0 and count (the DC/Nyquist
     pass): a 2-lane gather of the two 128-bit slots (plain, conjugated,
     or the backward's bin gather) and per-lane 128-bit stores. *)
  | AZinMir of int (* m: zin [2*(m*Ls  - k - (per-1))], reversed + conj   *)
  | AZoutMir of int (* m: zout[2*(m*OLs - k - (per-1))], reversed + conj   *)
  | AZoutTurnMir of int * int (* (l, c): zout[2*((Ls - k - c)*OLs + l)], conj *)
  | AZinSpec of int (* c: [zin[2*(c*Ls)] | zin[2*(c*Ls + count)]]           *)
  | AZinSpecC of int (* the same, conjugated                                *)
  | AZinSpecB of int * int (* (R, q): the c2r special leg q of radix R      *)
  | AZoutSpec of int * int (* (q, lane): zout[2*(q*OLs [+ count])], 128-bit *)
  | AZoutSpecT of int * int (* (c, which): zout[2*(c)] / zout[2*(count*OLs + c)] *)
  (* ── the real leaf r2z (real_il.ml): REAL lanes, no factor 2 in the
     address; the t2m mid's packed DC/Nyquist gather ── *)
  | AXinLeg of int (* l: zin [(size_t)l*Ls + k]   vec_width real columns   *)
  | AXoutLeg of int (* l: zout[(size_t)l*OLs + k]                            *)
  | AZinLegOff of int * int (* (l, off): zin[2*((size_t)l*Ls + k + off)]     *)
  | AZinSpecP of int (* c: the packed slot zin[2*(c*Ls)] = (a, b) as [a 0 | b 0] *)
  (* ── the real MONO rn1 (c2c_il.ml, 2026-09-30): the whole small real
     transform as one n1 body. REAL input lanes, (x, 0) per column; the
     backward's Hermitian half input (bin r-l conjugated) and REAL output
     lanes. No factor 2 in the real addresses. ── *)
  | AZinReal of int (* l: zin[(size_t)l*Ls + k] -> (x, 0) per column        *)
  | AZinHerm of int * int (* (l, r): conj of bin r-l, zin[2*((size_t)(r-l)*Ls + k)] *)
  | AZoutReal of int (* l: zout[(size_t)l*OLs + k], the real lane per column   *)
  (* ── the real FLAT leaf r1c (real_il.ml, 2026-09-30): digit p >= 1 of a
     column is a complex value, and the digit's run of `count` columns is
     block p of the c2c flat plane: complex p*pitch + k. off = the double
     offset of the vector inside the group's columns. ── *)
  | ADigOut of int * int (* (p, off): zout[(size_t)2p*OLs + 2*k + off] *)
  | ADigIn of int * int (* (p, off): zin [(size_t)2p*Ls  + 2*k + off] *)
  (* ── the real ROWS kind r2zr (real_il.ml, 2026-10-01): the real leaf over
     ROW-MAJOR rows. A lane is a ROW: row k+r's samples off.. load as one
     vector (Ls = the row pitch in doubles) and a block of vec_width such
     loads is transposed into the per-sample lane vectors. AZeroV is the
     zero vector (the imaginary lane of the real DC / Nyquist slots): no
     address, a register xor. ── *)
  | AXinRow of int * int (* (r, off): zin[((size_t)k + r)*Ls + off] *)
  | AZeroV

type cx_kind =
  | CIn of int (* input leg i (a packed-complex load) *)
  | CLoad of caddr
  (* a load with its ADDRESS in the DAG — the complete-IR replacement for
     CIn + the hand load edge. Same is_load/latency treatment as CIn. *)
  | CStore of caddr * t
  (* a store node: address + the value it sinks. First-class so the
     scheduler CAN see stores (Node.is_store, the B2 hook) — whether it
     SCHEDULES them is the placement policy's choice, not the IR's. *)
  | CUnpack of t * t * bool
  (* the 64-bit interleave within each 128-bit lane: lo (false) = unpacklo_pd
     [a0 b0 a2 b2], hi (true) = unpackhi_pd [a1 b1 a3 b3]. With CTurn it is
     the real leaf's store edge: four columns' (re, im) vectors -> the
     interleaved slots of each column (real_il.ml). *)
  | CTurn of t * t * bool
  (* COMPLEX-LANE DEINTERLEAVE of two vectors (the corner-turn round):
       even (false): [a0,a2,..,b0,b2,..]   odd (true): [a1,a3,..,b1,b3,..]
     (lane = one complex = 128 bits). log2(per) rounds of it transpose a
     per x per complex block (Cx_math.turn_transpose). ISA-parametric via
     Isa.cx_deint_pd: avx2 permute2f128 0x20/0x31, avx512 shuffle_f64x2
     0x88/0xDD, sse2 identity. *)
  | CPart of t * int
  (* complex lane c as a 128-bit value (Isa.cx_part_pd): avx2
     castpd256_pd128 / extractf128(.,1); avx512 castpd512_pd128 /
     extractf64x2(.,c). The odd-leg and leg-strided scatter quarters. *)
  | CAdd of t * t
  | CSub of t * t
  | CNeg of t
  (* -x : negate BOTH lanes (complex negation). The algebraic atom the
     rewrite passes need (mirrors Ir's NK_Neg): dedup_sub_pairs rewrites
     Sub(b,a) into Neg(Sub(a,b)) so mirrored subtractions share one node.
     Never constructed by the math builders — pass-introduced only, so its
     absence keeps every existing kernel byte-identical. *)
  | CRotNI of t (* x * (-i) : cflip then negate the IM lane *)
  | CRotPI of t
  (* x * (+i) : cflip then negate the RE lane. The backward twin of
     CRotNI — an inverse transform's quarter-turn goes the other way.
     (a+bi)*(+i) = -b + ai, so from cflip = [b,a] we negate lane 0. *)
  | CRotAdd of t * t
  (* a + i*y in ONE fused step: AVX2/SSE2 render = addsub(a, cflip y) —
     one shuffle + one vaddsubpd, no mask, legal in FWD kernels. Introduced
     for the wing construction's +i-side combines (the cadd/crot
     composition costs one extra uop). Never built by classic math paths,
     so flag-off emissions stay byte-identical. *)
  | CFmaC of float * t * t (* c*x + e,  real scalar c *)
  | CFnmaC of float * t * t (* -c*x + e, real scalar c *)
  | CTwC of float * float * t (* x * (c + i*s), emit-time constants *)
  | CTwV of (float * float) array * t
  (* x * w, with a DIFFERENT emit-time constant per complex lane. The K=1
     fused kernel needs this: one vector holds two DIFFERENT output indices
     k1 of the same transform, so their twiddles w_N^{k1*j2} differ. Still
     zero runtime cost — the whole [c,c,c',c'][-s,+s,-s',+s'] pair is a
     file-scope VLIT, exactly like CTwC's. Array length = vec_width/2. *)
  | CTwL of int * t
(* x * w[leg], w LOADED from the streamed VTW2 table — the bailey2 t2
     mid. Same BYTW2 shape as CTwC, but cvec/svec come from the runtime
     cursor `twp` instead of file-scope VLIT constants. The int is the
     LEG index; the record offset is (leg-1)*2*VW because leg 0 is
     untwiddled and each record is [c×VW][s×VW] (cos-first, sign-folded
     — one data-side shuffle, zero table-side work). *)

and t =
  { tag : int
  ; node : cx_kind
  }

(* Hash-consing: structural equality becomes tag equality, which is what
 * gives us CSE for free (the shared ±i rotations and repeated subsums in a
 * radix-8 body dedup automatically). Mirrors Ir.hashcons. *)
(* ── M12a: THE PER-EMISSION CONTEXT ──
   The five emission-POLICY cells that lived here as module refs (tw_log3,
   tw_pre, st_turn, st_turn_gs, mono_spill_slots) plus the cx_math/
   cx_render env knobs are now ONE record, created per emission by the
   driver (C2c_il.emit / emit_k1) and threaded to the readers.  The old
   refs were set by the driver and NEVER reset — harmless in one-shot
   gen_radix, a leak the moment cil enters the warm gen_set process (the
   M12a precondition for the corpus entry).  Field semantics unchanged:
   tw_log3 = VTW2 sourcing for the CTwL renderer; tw_pre = pre-twiddle on
   a backward T2 (T2P); st_turn = corner-turned store (T2T); st_turn_gs =
   leg-strided turned store (T2TG, implies st_turn); mono_spill_slots =
   Belady S[] slots for the current MONO codelet (mutable — set mid-
   emission once the spill plan exists).  tangent / w32_combine /
   wing_enabled / rotfma capture their VFFT_CX_* envs at ctx creation
   (the kernel Knobs snapshot intent; tangent also ORs the --cil-tangent
   CLI flag the driver passes). *)
type ctx =
  { tw_log3 : bool
  ; tw_group : bool
    (* t2c: CTwL sources from the _wc/_ws names a GROUP prologue binds
       (per-(d,leg) records hoisted out of the column loop — the z-T1S
       broadcast strategy, il_native_design.md §6c). Same naming as log3,
       no derivation; the two are mutually exclusive by construction. *)
  ; tw_pre : bool
  ; tw_gen2 : bool
    (* gen2 (2026-09-04): the twiddle stream is GENERATED — the W^1 pair
       record is the product of a per-pair T1 record (tw_re, cursor) and a
       per-call T2 broadcast record (tw_im, hoisted), and every higher leg
       is derived in-kernel by the PowW1 squaring tree on records. No
       N-sized table: the flat chain's tail stages read ~2*sqrt(N). *)
  ; colstride : bool
    (* t2cs (2026-09-04): the COLUMN-STRIDE form of t2 — a "column" is one
       BLOCK of the flat mixed-radix chain's short-run tail (D < vw): lane j
       of a vector comes from block k+j, so every leg load/store is two
       128-bit halves at stride Gs (in) / OGs (out); per-pair twiddle
       records (adjacent blocks carry different twiddles). The two-group
       "arrange halves" kernel a generic-N engine runs its tail on. *)
  ; st_turn : bool
  ; st_turn_gs : bool
  ; mutable mono_spill_slots : int
  ; tangent : bool
  ; w32_combine : bool
  ; wing_enabled : bool
  ; rotfma : bool
  }

let make_ctx ~tw_group ~tw_log3 ~tw_pre ~tw_gen2 ~colstride ~st_turn ~st_turn_gs ~tangent =
  { tw_log3
  ; tw_group
  ; tw_pre
  ; tw_gen2
  ; colstride
  ; st_turn
  ; st_turn_gs
  ; mono_spill_slots = 0
  ; tangent = tangent || Sys.getenv_opt "VFFT_CX_TANGENT" = Some "1"
  ; w32_combine = Sys.getenv_opt "VFFT_CX_W32TG" = Some "1"
  ; wing_enabled = Sys.getenv_opt "VFFT_CX_WING" = Some "1"
  ; rotfma = Sys.getenv_opt "VFFT_CX_ROTFMA" = Some "1"
  }
;;

let hcons : (cx_kind, t) Hashtbl.t = Hashtbl.create 256
let next_tag = ref 0

let reset () =
  Hashtbl.reset hcons;
  next_tag := 0
;;

let mk (nk : cx_kind) : t =
  match Hashtbl.find_opt hcons nk with
  | Some e -> e
  | None ->
    let e = { tag = !next_tag; node = nk } in
    incr next_tag;
    Hashtbl.add hcons nk e;
    e
;;

let cin i = mk (CIn i)
let cload a = mk (CLoad a)
let cstore a v = mk (CStore (a, v))
let cturn a b odd = mk (CTurn (a, b, odd))
let cunpack a b hi = mk (CUnpack (a, b, hi))
let cpart a c = mk (CPart (a, c))
let clo a = cpart a 0
let chi a = cpart a 1
let cadd a b = mk (CAdd (a, b))
let csub a b = mk (CSub (a, b))
let cneg a = mk (CNeg a)
let crot a = mk (CRotNI a)
let crotp a = mk (CRotPI a)
let crotadd a b = mk (CRotAdd (a, b))
let cfma c x e = mk (CFmaC (c, x, e))
let cfnma c x e = mk (CFnmaC (c, x, e))
let ctw c s x = mk (CTwC (c, s, x))
let ctwv w x = mk (CTwV (w, x))
let ctwl leg x = mk (CTwL (leg, x))
