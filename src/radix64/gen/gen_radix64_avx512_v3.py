#!/usr/bin/env python3
"""
gen_radix64_avx512_v3.py — Generate straight-line AVX-512 DFT-64 N1 kernels
                            (forward AND backward) using 8×8 decomposition.

Changes from v2:
  - direction parameter ('fwd' or 'bwd') flows through radix-8 butterfly and twiddles
  - Generates two separate functions: _fwd_avx512() and _bwd_avx512()
  - Backward kernel is native — no conjugate trick, no temp buffers
  - Twiddle broadcasts hoisted above k-loop (loop-invariant)

Usage:
  python3 gen_radix64_avx512_v3.py > fft_radix64_avx512_n1_gen.h
"""

import math, sys

# ──────────────────────────────────────────────────────────────────
# W₆₄ twiddle values
# ──────────────────────────────────────────────────────────────────

def w64(e):
    """W₆₄^e = cos(2πe/64) - j·sin(2πe/64) (forward convention)"""
    e = e % 64
    angle = 2.0 * math.pi * e / 64.0
    return (math.cos(angle), -math.sin(angle))

def w64_label(e):
    return f"W64_{e % 64}"

def twiddle_is_trivial(e):
    """Returns (is_special, type) for W₆₄^e"""
    e = e % 64
    if e == 0:  return True, 'one'
    if e == 16: return True, 'neg_j'
    if e == 32: return True, 'neg_one'
    if e == 48: return True, 'pos_j'
    if e == 8:  return True, 'w8_1'
    if e == 24: return True, 'w8_3'
    if e == 40: return True, 'neg_w8_1'
    if e == 56: return True, 'neg_w8_3'
    return False, 'cmul'

# ──────────────────────────────────────────────────────────────────
# Code emitter
# ──────────────────────────────────────────────────────────────────

class Emitter:
    def __init__(self):
        self.lines = []
        self.indent = 1
        self.spill_slot = 0
        self.spill_count = 0
        self.reload_count = 0
        self.twiddles_needed = set()
    
    def emit(self, line=""):
        self.lines.append("    " * self.indent + line)
    
    def comment(self, text):
        self.emit(f"/* {text} */")
    
    def blank(self):
        self.lines.append("")
    
    def alloc_spill(self, count):
        first = self.spill_slot
        self.spill_slot += count
        return first
    
    def emit_spill(self, var, slot):
        self.emit(f"_mm512_store_pd(&spill_re[{slot} * 8], {var}_re);")
        self.emit(f"_mm512_store_pd(&spill_im[{slot} * 8], {var}_im);")
        self.spill_count += 1
    
    def emit_reload(self, var, slot):
        self.emit(f"{var}_re = _mm512_load_pd(&spill_re[{slot} * 8]);")
        self.emit(f"{var}_im = _mm512_load_pd(&spill_im[{slot} * 8]);")
        self.reload_count += 1
    
    def emit_load_input(self, var, n):
        self.emit(f"{var}_re = R64G_LD(&in_re[{n} * K + k]);")
        self.emit(f"{var}_im = R64G_LD(&in_im[{n} * K + k]);")
    
    def emit_store_output(self, var, m):
        self.emit(f"R64G_ST(&out_re[{m} * K + k], {var}_re);")
        self.emit(f"R64G_ST(&out_im[{m} * K + k], {var}_im);")
    
    # ──────────────────────────────────────────────────────────────
    # Radix-8 butterfly (direction-aware)
    # ──────────────────────────────────────────────────────────────
    
    def emit_radix8_n1(self, x_prefix, direction, comment_str=""):
        """
        Emit inline radix-8 N1 butterfly, forward or backward.
        
        Structure:
          radix-4 on {x0,x2,x4,x6} → even
          radix-4 on {x1,x3,x5,x7} → odd
          W₈ apply on odd (direction-dependent)
          combine: y[i] = even[i] ± odd[i]
        
        Backward differs in:
          1. Radix-4: j-rotation is +j instead of -j
          2. W₈ twiddles are conjugated: W₈¹→(1+j)/√2, W₈³→(-1+j)/√2, -j→+j
        """
        assert direction in ('fwd', 'bwd')
        fwd = (direction == 'fwd')
        p = x_prefix
        
        if comment_str:
            self.comment(f"{comment_str} [{direction}]")
        
        # Even radix-4: x0, x2, x4, x6
        self.comment(f"radix-4 even: {p}0,{p}2,{p}4,{p}6")
        self.emit(f"{{ __m512d e0r, e0i, e1r, e1i, e2r, e2i, e3r, e3i;")
        self.emit(f"  __m512d t0r, t0i, t1r, t1i, t2r, t2i, t3r, t3i;")
        # Level 1: same for fwd and bwd
        self.emit(f"  t0r = _mm512_add_pd({p}0_re, {p}4_re); t0i = _mm512_add_pd({p}0_im, {p}4_im);")
        self.emit(f"  t1r = _mm512_sub_pd({p}0_re, {p}4_re); t1i = _mm512_sub_pd({p}0_im, {p}4_im);")
        self.emit(f"  t2r = _mm512_add_pd({p}2_re, {p}6_re); t2i = _mm512_add_pd({p}2_im, {p}6_im);")
        self.emit(f"  t3r = _mm512_sub_pd({p}2_re, {p}6_re); t3i = _mm512_sub_pd({p}2_im, {p}6_im);")
        # Level 2: direction-dependent j-rotation on t3
        self.emit(f"  e0r = _mm512_add_pd(t0r, t2r); e0i = _mm512_add_pd(t0i, t2i);")
        self.emit(f"  e2r = _mm512_sub_pd(t0r, t2r); e2i = _mm512_sub_pd(t0i, t2i);")
        if fwd:
            # -j·t3 = (t3i, -t3r)
            self.emit(f"  e1r = _mm512_add_pd(t1r, t3i); e1i = _mm512_sub_pd(t1i, t3r);")
            self.emit(f"  e3r = _mm512_sub_pd(t1r, t3i); e3i = _mm512_add_pd(t1i, t3r);")
        else:
            # +j·t3 = (-t3i, t3r)
            self.emit(f"  e1r = _mm512_sub_pd(t1r, t3i); e1i = _mm512_add_pd(t1i, t3r);")
            self.emit(f"  e3r = _mm512_add_pd(t1r, t3i); e3i = _mm512_sub_pd(t1i, t3r);")
        
        # Odd radix-4: x1, x3, x5, x7
        self.comment(f"radix-4 odd: {p}1,{p}3,{p}5,{p}7")
        self.emit(f"  __m512d o0r, o0i, o1r, o1i, o2r, o2i, o3r, o3i;")
        self.emit(f"  t0r = _mm512_add_pd({p}1_re, {p}5_re); t0i = _mm512_add_pd({p}1_im, {p}5_im);")
        self.emit(f"  t1r = _mm512_sub_pd({p}1_re, {p}5_re); t1i = _mm512_sub_pd({p}1_im, {p}5_im);")
        self.emit(f"  t2r = _mm512_add_pd({p}3_re, {p}7_re); t2i = _mm512_add_pd({p}3_im, {p}7_im);")
        self.emit(f"  t3r = _mm512_sub_pd({p}3_re, {p}7_re); t3i = _mm512_sub_pd({p}3_im, {p}7_im);")
        self.emit(f"  o0r = _mm512_add_pd(t0r, t2r); o0i = _mm512_add_pd(t0i, t2i);")
        self.emit(f"  o2r = _mm512_sub_pd(t0r, t2r); o2i = _mm512_sub_pd(t0i, t2i);")
        if fwd:
            self.emit(f"  o1r = _mm512_add_pd(t1r, t3i); o1i = _mm512_sub_pd(t1i, t3r);")
            self.emit(f"  o3r = _mm512_sub_pd(t1r, t3i); o3i = _mm512_add_pd(t1i, t3r);")
        else:
            self.emit(f"  o1r = _mm512_sub_pd(t1r, t3i); o1i = _mm512_add_pd(t1i, t3r);")
            self.emit(f"  o3r = _mm512_add_pd(t1r, t3i); o3i = _mm512_sub_pd(t1i, t3r);")
        
        # W₈ apply on odd outputs (direction-dependent)
        if fwd:
            self.comment("W₈ apply [fwd]: o1 *= W₈¹, o2 *= -j, o3 *= W₈³")
            # W₈¹ = (1-j)/√2: re' = (re+im)/√2, im' = (im-re)/√2
            self.emit(f"  t0r = _mm512_mul_pd(_mm512_add_pd(o1r, o1i), sqrt2_inv);")
            self.emit(f"  t0i = _mm512_mul_pd(_mm512_sub_pd(o1i, o1r), sqrt2_inv);")
            self.emit(f"  o1r = t0r; o1i = t0i;")
            # -j: (im, -re)
            self.emit(f"  t0r = o2i; t0i = _mm512_xor_pd(o2r, sign_flip);")
            self.emit(f"  o2r = t0r; o2i = t0i;")
            # W₈³ = -(1+j)/√2: re' = (im-re)/√2, im' = -(re+im)/√2
            self.emit(f"  t0r = _mm512_mul_pd(_mm512_sub_pd(o3i, o3r), sqrt2_inv);")
            self.emit(f"  t0i = _mm512_xor_pd(_mm512_mul_pd(_mm512_add_pd(o3r, o3i), sqrt2_inv), sign_flip);")
            self.emit(f"  o3r = t0r; o3i = t0i;")
        else:
            self.comment("W₈ apply [bwd]: o1 *= conj(W₈¹), o2 *= +j, o3 *= conj(W₈³)")
            # conj(W₈¹) = (1+j)/√2: re' = (re-im)/√2, im' = (re+im)/√2
            self.emit(f"  t0r = _mm512_mul_pd(_mm512_sub_pd(o1r, o1i), sqrt2_inv);")
            self.emit(f"  t0i = _mm512_mul_pd(_mm512_add_pd(o1r, o1i), sqrt2_inv);")
            self.emit(f"  o1r = t0r; o1i = t0i;")
            # +j: (-im, re)
            self.emit(f"  t0r = _mm512_xor_pd(o2i, sign_flip); t0i = o2r;")
            self.emit(f"  o2r = t0r; o2i = t0i;")
            # conj(W₈³) = (-1+j)/√2: re' = -(re+im)/√2, im' = (re-im)/√2
            self.emit(f"  t0r = _mm512_xor_pd(_mm512_mul_pd(_mm512_add_pd(o3r, o3i), sqrt2_inv), sign_flip);")
            self.emit(f"  t0i = _mm512_mul_pd(_mm512_sub_pd(o3r, o3i), sqrt2_inv);")
            self.emit(f"  o3r = t0r; o3i = t0i;")
        
        # Combine: y[i] = even[i] + odd[i], y[i+4] = even[i] - odd[i]
        # (same for fwd and bwd)
        self.comment("combine: y = even ± odd")
        self.emit(f"  {p}0_re = _mm512_add_pd(e0r, o0r); {p}0_im = _mm512_add_pd(e0i, o0i);")
        self.emit(f"  {p}4_re = _mm512_sub_pd(e0r, o0r); {p}4_im = _mm512_sub_pd(e0i, o0i);")
        self.emit(f"  {p}1_re = _mm512_add_pd(e1r, o1r); {p}1_im = _mm512_add_pd(e1i, o1i);")
        self.emit(f"  {p}5_re = _mm512_sub_pd(e1r, o1r); {p}5_im = _mm512_sub_pd(e1i, o1i);")
        self.emit(f"  {p}2_re = _mm512_add_pd(e2r, o2r); {p}2_im = _mm512_add_pd(e2i, o2i);")
        self.emit(f"  {p}6_re = _mm512_sub_pd(e2r, o2r); {p}6_im = _mm512_sub_pd(e2i, o2i);")
        self.emit(f"  {p}3_re = _mm512_add_pd(e3r, o3r); {p}3_im = _mm512_add_pd(e3i, o3i);")
        self.emit(f"  {p}7_re = _mm512_sub_pd(e3r, o3r); {p}7_im = _mm512_sub_pd(e3i, o3i);")
        self.emit(f"}}")
    
    # ──────────────────────────────────────────────────────────────
    # W₆₄ twiddle apply (direction-aware)
    # ──────────────────────────────────────────────────────────────
    
    def emit_twiddle(self, dst, src, e64, direction):
        """Apply W₆₄^e64 (fwd) or conj(W₆₄^e64) (bwd) to src, store in dst."""
        assert direction in ('fwd', 'bwd')
        fwd = (direction == 'fwd')
        is_special, typ = twiddle_is_trivial(e64)
        
        if typ == 'one':
            # W=1, conj(1)=1
            if dst != src:
                self.emit(f"{dst}_re = {src}_re; {dst}_im = {src}_im;")
        elif typ == 'neg_one':
            # W=-1, conj(-1)=-1
            self.emit(f"{dst}_re = _mm512_xor_pd({src}_re, sign_flip); {dst}_im = _mm512_xor_pd({src}_im, sign_flip);")
        elif typ == 'neg_j':
            if fwd:
                # -j: (im, -re)
                self.emit(f"{{ const __m512d t = {src}_re;")
                self.emit(f"  {dst}_re = {src}_im;")
                self.emit(f"  {dst}_im = _mm512_xor_pd(t, sign_flip); }}")
            else:
                # conj(-j) = +j: (-im, re)
                self.emit(f"{{ const __m512d t = {src}_re;")
                self.emit(f"  {dst}_re = _mm512_xor_pd({src}_im, sign_flip);")
                self.emit(f"  {dst}_im = t; }}")
        elif typ == 'pos_j':
            if fwd:
                # +j: (-im, re)
                self.emit(f"{{ const __m512d t = {src}_re;")
                self.emit(f"  {dst}_re = _mm512_xor_pd({src}_im, sign_flip);")
                self.emit(f"  {dst}_im = t; }}")
            else:
                # conj(+j) = -j: (im, -re)
                self.emit(f"{{ const __m512d t = {src}_re;")
                self.emit(f"  {dst}_re = {src}_im;")
                self.emit(f"  {dst}_im = _mm512_xor_pd(t, sign_flip); }}")
        elif typ == 'w8_1':
            if fwd:
                # W₈¹ = (1-j)/√2: re'=(re+im)/√2, im'=(im-re)/√2
                self.emit(f"{{ const __m512d t_re = {src}_re; const __m512d t_im = {src}_im;")
                self.emit(f"  {dst}_re = _mm512_mul_pd(_mm512_add_pd(t_re, t_im), sqrt2_inv);")
                self.emit(f"  {dst}_im = _mm512_mul_pd(_mm512_sub_pd(t_im, t_re), sqrt2_inv); }}")
            else:
                # conj(W₈¹) = (1+j)/√2: re'=(re-im)/√2, im'=(re+im)/√2
                self.emit(f"{{ const __m512d t_re = {src}_re; const __m512d t_im = {src}_im;")
                self.emit(f"  {dst}_re = _mm512_mul_pd(_mm512_sub_pd(t_re, t_im), sqrt2_inv);")
                self.emit(f"  {dst}_im = _mm512_mul_pd(_mm512_add_pd(t_re, t_im), sqrt2_inv); }}")
        elif typ == 'w8_3':
            if fwd:
                # W₈³ = -(1+j)/√2: re'=(im-re)/√2, im'=-(re+im)/√2
                self.emit(f"{{ const __m512d t_re = {src}_re; const __m512d t_im = {src}_im;")
                self.emit(f"  {dst}_re = _mm512_mul_pd(_mm512_sub_pd(t_im, t_re), sqrt2_inv);")
                self.emit(f"  {dst}_im = _mm512_xor_pd(_mm512_mul_pd(_mm512_add_pd(t_re, t_im), sqrt2_inv), sign_flip); }}")
            else:
                # conj(W₈³) = (-1+j)/√2: re'=-(re+im)/√2, im'=(re-im)/√2
                self.emit(f"{{ const __m512d t_re = {src}_re; const __m512d t_im = {src}_im;")
                self.emit(f"  {dst}_re = _mm512_xor_pd(_mm512_mul_pd(_mm512_add_pd(t_re, t_im), sqrt2_inv), sign_flip);")
                self.emit(f"  {dst}_im = _mm512_mul_pd(_mm512_sub_pd(t_re, t_im), sqrt2_inv); }}")
        elif typ == 'neg_w8_1':
            if fwd:
                # -W₈¹ = -(1-j)/√2: re'=-(re+im)/√2, im'=(re-im)/√2
                self.emit(f"{{ const __m512d t_re = {src}_re; const __m512d t_im = {src}_im;")
                self.emit(f"  {dst}_re = _mm512_xor_pd(_mm512_mul_pd(_mm512_add_pd(t_re, t_im), sqrt2_inv), sign_flip);")
                self.emit(f"  {dst}_im = _mm512_mul_pd(_mm512_sub_pd(t_re, t_im), sqrt2_inv); }}")
            else:
                # conj(-W₈¹) = -(1+j)/√2: re'=(im-re)/√2, im'=-(re+im)/√2
                # (same as forward w8_3!)
                self.emit(f"{{ const __m512d t_re = {src}_re; const __m512d t_im = {src}_im;")
                self.emit(f"  {dst}_re = _mm512_mul_pd(_mm512_sub_pd(t_im, t_re), sqrt2_inv);")
                self.emit(f"  {dst}_im = _mm512_xor_pd(_mm512_mul_pd(_mm512_add_pd(t_re, t_im), sqrt2_inv), sign_flip); }}")
        elif typ == 'neg_w8_3':
            if fwd:
                # -W₈³ = (1+j)/√2: re'=(re-im)/√2, im'=(re+im)/√2
                self.emit(f"{{ const __m512d t_re = {src}_re; const __m512d t_im = {src}_im;")
                self.emit(f"  {dst}_re = _mm512_mul_pd(_mm512_sub_pd(t_re, t_im), sqrt2_inv);")
                self.emit(f"  {dst}_im = _mm512_mul_pd(_mm512_add_pd(t_re, t_im), sqrt2_inv); }}")
            else:
                # conj(-W₈³) = (1-j)/√2: re'=(re+im)/√2, im'=(im-re)/√2
                # (same as forward w8_1!)
                self.emit(f"{{ const __m512d t_re = {src}_re; const __m512d t_im = {src}_im;")
                self.emit(f"  {dst}_re = _mm512_mul_pd(_mm512_add_pd(t_re, t_im), sqrt2_inv);")
                self.emit(f"  {dst}_im = _mm512_mul_pd(_mm512_sub_pd(t_im, t_re), sqrt2_inv); }}")
        else:
            # General cmul: fwd uses (wr, wi), bwd uses (wr, -wi)
            self.twiddles_needed.add(e64)
            label = w64_label(e64)
            if fwd:
                self.emit(f"{{ const __m512d tmp_re = {src}_re;")
                self.emit(f"  {dst}_re = _mm512_fmsub_pd({src}_re, tw_{label}_re, _mm512_mul_pd({src}_im, tw_{label}_im));")
                self.emit(f"  {dst}_im = _mm512_fmadd_pd(tmp_re, tw_{label}_im, _mm512_mul_pd({src}_im, tw_{label}_re));")
                self.emit(f"}}")
            else:
                # conj twiddle: (wr, -wi) → re' = re*wr + im*wi, im' = im*wr - re*wi
                self.emit(f"{{ const __m512d tmp_re = {src}_re;")
                self.emit(f"  {dst}_re = _mm512_fmadd_pd({src}_re, tw_{label}_re, _mm512_mul_pd({src}_im, tw_{label}_im));")
                self.emit(f"  {dst}_im = _mm512_fmsub_pd({src}_im, tw_{label}_re, _mm512_mul_pd(tmp_re, tw_{label}_im));")
                self.emit(f"}}")

# ──────────────────────────────────────────────────────────────────
# Kernel generation (parameterized by direction)
# ──────────────────────────────────────────────────────────────────

def emit_kernel(em, direction, tw_needed, base_slot):
    """Emit one kernel function (fwd or bwd) into the emitter."""
    dir_label = direction  # 'fwd' or 'bwd'
    
    em.lines.append(f"static TARGET_AVX512 void")
    em.lines.append(f"radix64_n1_dit_kernel_{dir_label}_avx512(")
    em.lines.append(f"    const double *RESTRICT in_re, const double *RESTRICT in_im,")
    em.lines.append(f"    double *RESTRICT out_re, double *RESTRICT out_im,")
    em.lines.append(f"    size_t K)")
    em.lines.append(f"{{")
    
    em.indent = 1
    em.spill_count = 0
    em.reload_count = 0
    
    # Constants
    em.emit("const __m512d sign_flip = _mm512_set1_pd(-0.0);")
    em.emit("const __m512d sqrt2_inv = _mm512_set1_pd(0.70710678118654752440);")
    em.blank()
    
    # Spill buffer
    em.emit(f"ALIGNAS_64 double spill_re[64 * 8];  /* 4096 bytes */")
    em.emit(f"ALIGNAS_64 double spill_im[64 * 8];")
    em.blank()
    
    # Working registers
    em.emit("__m512d x0_re, x0_im, x1_re, x1_im, x2_re, x2_im, x3_re, x3_im;")
    em.emit("__m512d x4_re, x4_im, x5_re, x5_im, x6_re, x6_im, x7_re, x7_im;")
    em.blank()
    
    # Hoisted twiddle broadcasts
    em.comment(f"Pre-broadcast twiddle constants [{dir_label}]")
    for e in sorted(tw_needed):
        label = w64_label(e)
        em.emit(f"const __m512d tw_{label}_re = _mm512_set1_pd({label}_re);")
        em.emit(f"const __m512d tw_{label}_im = _mm512_set1_pd({label}_im);")
    em.blank()
    
    # k-loop
    em.emit("for (size_t k = 0; k < K; k += 8) {")
    em.indent += 1
    
    # ── PASS 1: 8 sub-FFTs ──
    em.comment("═══════════════════════════════════════════════════")
    em.comment(f"PASS 1: 8 radix-8 sub-FFTs → spill [{dir_label}]")
    em.comment("═══════════════════════════════════════════════════")
    em.blank()
    
    for n2 in range(8):
        em.comment(f"── Sub-FFT n₂={n2} ──")
        for n1 in range(8):
            idx = 8 * n1 + n2
            em.emit_load_input(f"x{n1}", idx)
        em.blank()
        
        em.emit_radix8_n1("x", direction, f"radix-8 sub-FFT n₂={n2}")
        em.blank()
        
        em.comment(f"spill sub-FFT {n2}")
        for k1 in range(8):
            em.emit_spill(f"x{k1}", base_slot + n2 * 8 + k1)
        em.blank()
    
    # ── PASS 2: 8 column combines ──
    em.comment("═══════════════════════════════════════════════════")
    em.comment(f"PASS 2: twiddles + 8 radix-8 combines → output [{dir_label}]")
    em.comment("═══════════════════════════════════════════════════")
    em.blank()
    
    for k1 in range(8):
        em.comment(f"── Column k₁={k1} ──")
        for n2 in range(8):
            em.emit_reload(f"x{n2}", base_slot + n2 * 8 + k1)
        em.blank()
        
        if k1 > 0:
            em.comment(f"W₆₄ twiddles for column k₁={k1} [{dir_label}]")
            for n2 in range(1, 8):
                e = (n2 * k1) % 64
                em.emit_twiddle(f"x{n2}", f"x{n2}", e, direction)
            em.blank()
        
        em.emit_radix8_n1("x", direction, f"radix-8 combine column k₁={k1}")
        em.blank()
        
        em.comment(f"store column {k1}")
        for k2 in range(8):
            em.emit_store_output(f"x{k2}", k1 + 8 * k2)
        em.blank()
    
    # Close k-loop
    em.indent -= 1
    em.emit("} /* end k-loop */")
    
    em.lines.append("}")
    em.lines.append("")
    
    return em.spill_count, em.reload_count

# ──────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────

def generate():
    em = Emitter()
    base_slot = em.alloc_spill(64)
    
    # Collect twiddle constants needed
    tw_needed = set()
    for n2 in range(1, 8):
        for k1 in range(1, 8):
            e = (n2 * k1) % 64
            is_special, _ = twiddle_is_trivial(e)
            if not is_special:
                tw_needed.add(e)
    
    # ── File header ──
    em.lines.append("/**")
    em.lines.append(" * @file fft_radix64_avx512_n1_gen.h")
    em.lines.append(" * @brief GENERATED DFT-64 AVX-512 kernels (forward + backward)")
    em.lines.append(" *")
    em.lines.append(" * Auto-generated by gen_radix64_avx512_v3.py.")
    em.lines.append(" * 8×8 Cooley-Tukey, explicit spills, hoisted twiddle broadcasts.")
    em.lines.append(" * Forward and backward are native — no conjugate trick.")
    em.lines.append(" */")
    em.lines.append("")
    em.lines.append("#ifndef FFT_RADIX64_AVX512_N1_GEN_H")
    em.lines.append("#define FFT_RADIX64_AVX512_N1_GEN_H")
    em.lines.append("")
    em.lines.append("#include <immintrin.h>")
    em.lines.append("")
    
    # Twiddle scalar constants (shared by both directions — same re, same im)
    em.lines.append("/* ── W₆₄ twiddle constants (guarded for multi-include) ── */")
    em.lines.append("#ifndef FFT_RADIX64_GEN_TWIDDLES_DEFINED")
    em.lines.append("#define FFT_RADIX64_GEN_TWIDDLES_DEFINED")
    for e in sorted(tw_needed):
        wr, wi = w64(e)
        em.lines.append(f"static const double {w64_label(e)}_re = {wr:.20e};")
        em.lines.append(f"static const double {w64_label(e)}_im = {wi:.20e};")
    em.lines.append("#endif /* FFT_RADIX64_GEN_TWIDDLES_DEFINED */")
    em.lines.append("")
    
    # Load/store macros
    em.lines.append("#ifndef R64G_LD")
    em.lines.append("#define R64G_LD(p) _mm512_loadu_pd(p)")
    em.lines.append("#endif")
    em.lines.append("#ifndef R64G_ST")
    em.lines.append("#define R64G_ST(p,v) _mm512_storeu_pd((p),(v))")
    em.lines.append("#endif")
    em.lines.append("")
    
    # ── Forward kernel ──
    fwd_spills, fwd_reloads = emit_kernel(em, 'fwd', tw_needed, base_slot)
    
    # ── Backward kernel ──
    bwd_spills, bwd_reloads = emit_kernel(em, 'bwd', tw_needed, base_slot)
    
    em.lines.append("#endif /* FFT_RADIX64_AVX512_N1_GEN_H */")
    
    return em, tw_needed, fwd_spills, fwd_reloads, bwd_spills, bwd_reloads

def main():
    em, tw_needed, fwd_spills, fwd_reloads, bwd_spills, bwd_reloads = generate()
    
    print("\n".join(em.lines))
    
    print(f"\n=== CODEGEN STATS ===", file=sys.stderr)
    print(f"Forward:  {fwd_spills} spills + {fwd_reloads} reloads = {fwd_spills+fwd_reloads} L1 ops", file=sys.stderr)
    print(f"Backward: {bwd_spills} spills + {bwd_reloads} reloads = {bwd_spills+bwd_reloads} L1 ops", file=sys.stderr)
    print(f"Unique generic twiddles: {len(tw_needed)}", file=sys.stderr)
    print(f"Generated lines: {len(em.lines)}", file=sys.stderr)

if __name__ == '__main__':
    main()
