#!/usr/bin/env python3
"""
gen_radix128_avx512.py — Generate straight-line AVX-512 DFT-128 N1 kernels
                          (forward AND backward) using 16×8 decomposition.

Structure:
  Pass 1: 8 radix-16 sub-FFTs (inline 4×4) → spill 128 slots
  Pass 2: 16 radix-8 column combines with W₁₂₈ twiddles → output

  Spill: 128 stores + 128 loads = 256 L1 ops, 16 KB buffer

Usage:
  python3 gen_radix128_avx512.py > fft_radix128_avx512_n1_gen.h
"""

import math, sys

# ──────────────────────────────────────────────────────────────────
# Twiddle helpers — generalized for any N
# ──────────────────────────────────────────────────────────────────

def wN(e, N):
    """W_N^e = cos(2πe/N) - j·sin(2πe/N) (forward convention)"""
    e = e % N
    angle = 2.0 * math.pi * e / N
    return (math.cos(angle), -math.sin(angle))

def wN_label(e, N):
    return f"W{N}_{e % N}"

def twiddle_is_trivial(e, N):
    """Check if W_N^e maps to 8th root of unity (special case)."""
    e = e % N
    if e == 0:
        return True, 'one'
    # Check if 8*e/N is an integer → maps to an 8th root
    if (8 * e) % N == 0:
        octant = (8 * e) // N
        types = ['one', 'w8_1', 'neg_j', 'w8_3',
                 'neg_one', 'neg_w8_1', 'pos_j', 'neg_w8_3']
        return True, types[octant % 8]
    return False, 'cmul'

# ──────────────────────────────────────────────────────────────────
# Code emitter
# ──────────────────────────────────────────────────────────────────

class Emitter:
    def __init__(self, N):
        self.N = N
        self.lines = []
        self.indent = 1
        self.spill_count = 0
        self.reload_count = 0
        self.twiddles_needed = set()  # set of (e, N) tuples
    
    def emit(self, line=""):
        self.lines.append("    " * self.indent + line)
    
    def comment(self, text):
        self.emit(f"/* {text} */")
    
    def blank(self):
        self.lines.append("")
    
    def emit_spill(self, var, slot):
        self.emit(f"_mm512_store_pd(&spill_re[{slot} * 8], {var}_re);")
        self.emit(f"_mm512_store_pd(&spill_im[{slot} * 8], {var}_im);")
        self.spill_count += 1
    
    def emit_reload(self, var, slot):
        self.emit(f"{var}_re = _mm512_load_pd(&spill_re[{slot} * 8]);")
        self.emit(f"{var}_im = _mm512_load_pd(&spill_im[{slot} * 8]);")
        self.reload_count += 1
    
    def emit_load_input(self, var, n):
        self.emit(f"{var}_re = R128G_LD(&in_re[{n} * K + k]);")
        self.emit(f"{var}_im = R128G_LD(&in_im[{n} * K + k]);")
    
    def emit_store_output(self, var, m):
        self.emit(f"R128G_ST(&out_re[{m} * K + k], {var}_re);")
        self.emit(f"R128G_ST(&out_im[{m} * K + k], {var}_im);")
    
    # ──────────────────────────────────────────────────────────
    # Radix-4 butterfly (building block)
    # ──────────────────────────────────────────────────────────
    
    def emit_radix4_inline(self, v, direction):
        """
        Inline radix-4 DIT on 4 variables v[0]..v[3] (in-place).
        v is a list of 4 variable name prefixes.
        
        y[0] = x0+x1+x2+x3
        y[1] = x0 ∓j·x1 - x2 ±j·x3    (∓ for fwd, ± for bwd)
        y[2] = x0-x1+x2-x3
        y[3] = x0 ±j·x1 - x2 ∓j·x3
        """
        fwd = (direction == 'fwd')
        a, b, c, d = v[0], v[1], v[2], v[3]
        
        self.emit(f"{{ __m512d t0r, t0i, t1r, t1i, t2r, t2i, t3r, t3i;")
        self.emit(f"  t0r = _mm512_add_pd({a}_re, {c}_re); t0i = _mm512_add_pd({a}_im, {c}_im);")
        self.emit(f"  t1r = _mm512_sub_pd({a}_re, {c}_re); t1i = _mm512_sub_pd({a}_im, {c}_im);")
        self.emit(f"  t2r = _mm512_add_pd({b}_re, {d}_re); t2i = _mm512_add_pd({b}_im, {d}_im);")
        self.emit(f"  t3r = _mm512_sub_pd({b}_re, {d}_re); t3i = _mm512_sub_pd({b}_im, {d}_im);")
        # y0 = t0+t2, y2 = t0-t2
        self.emit(f"  {a}_re = _mm512_add_pd(t0r, t2r); {a}_im = _mm512_add_pd(t0i, t2i);")
        self.emit(f"  {c}_re = _mm512_sub_pd(t0r, t2r); {c}_im = _mm512_sub_pd(t0i, t2i);")
        if fwd:
            # y1 = t1 - j·t3 = (t1r+t3i, t1i-t3r)
            self.emit(f"  {b}_re = _mm512_add_pd(t1r, t3i); {b}_im = _mm512_sub_pd(t1i, t3r);")
            self.emit(f"  {d}_re = _mm512_sub_pd(t1r, t3i); {d}_im = _mm512_add_pd(t1i, t3r);")
        else:
            # y1 = t1 + j·t3 = (t1r-t3i, t1i+t3r)
            self.emit(f"  {b}_re = _mm512_sub_pd(t1r, t3i); {b}_im = _mm512_add_pd(t1i, t3r);")
            self.emit(f"  {d}_re = _mm512_add_pd(t1r, t3i); {d}_im = _mm512_sub_pd(t1i, t3r);")
        self.emit(f"}}")
    
    # ──────────────────────────────────────────────────────────
    # Radix-8 butterfly (from v3, now with variable names)
    # ──────────────────────────────────────────────────────────
    
    def emit_radix8_inline(self, v, direction, comment_str=""):
        """
        Inline radix-8 DIT on 8 variables v[0]..v[7] (in-place).
        Decomposition: radix-4(evens) + radix-4(odds) + W₈ twiddles + combine.
        """
        assert len(v) == 8
        fwd = (direction == 'fwd')
        
        if comment_str:
            self.comment(f"{comment_str} [{direction}]")
        
        # Even radix-4: v[0], v[2], v[4], v[6]
        self.comment(f"radix-4 even: {v[0]},{v[2]},{v[4]},{v[6]}")
        self.emit(f"{{ __m512d e0r, e0i, e1r, e1i, e2r, e2i, e3r, e3i;")
        self.emit(f"  __m512d t0r, t0i, t1r, t1i, t2r, t2i, t3r, t3i;")
        self.emit(f"  t0r = _mm512_add_pd({v[0]}_re, {v[4]}_re); t0i = _mm512_add_pd({v[0]}_im, {v[4]}_im);")
        self.emit(f"  t1r = _mm512_sub_pd({v[0]}_re, {v[4]}_re); t1i = _mm512_sub_pd({v[0]}_im, {v[4]}_im);")
        self.emit(f"  t2r = _mm512_add_pd({v[2]}_re, {v[6]}_re); t2i = _mm512_add_pd({v[2]}_im, {v[6]}_im);")
        self.emit(f"  t3r = _mm512_sub_pd({v[2]}_re, {v[6]}_re); t3i = _mm512_sub_pd({v[2]}_im, {v[6]}_im);")
        self.emit(f"  e0r = _mm512_add_pd(t0r, t2r); e0i = _mm512_add_pd(t0i, t2i);")
        self.emit(f"  e2r = _mm512_sub_pd(t0r, t2r); e2i = _mm512_sub_pd(t0i, t2i);")
        if fwd:
            self.emit(f"  e1r = _mm512_add_pd(t1r, t3i); e1i = _mm512_sub_pd(t1i, t3r);")
            self.emit(f"  e3r = _mm512_sub_pd(t1r, t3i); e3i = _mm512_add_pd(t1i, t3r);")
        else:
            self.emit(f"  e1r = _mm512_sub_pd(t1r, t3i); e1i = _mm512_add_pd(t1i, t3r);")
            self.emit(f"  e3r = _mm512_add_pd(t1r, t3i); e3i = _mm512_sub_pd(t1i, t3r);")
        
        # Odd radix-4: v[1], v[3], v[5], v[7]
        self.comment(f"radix-4 odd: {v[1]},{v[3]},{v[5]},{v[7]}")
        self.emit(f"  __m512d o0r, o0i, o1r, o1i, o2r, o2i, o3r, o3i;")
        self.emit(f"  t0r = _mm512_add_pd({v[1]}_re, {v[5]}_re); t0i = _mm512_add_pd({v[1]}_im, {v[5]}_im);")
        self.emit(f"  t1r = _mm512_sub_pd({v[1]}_re, {v[5]}_re); t1i = _mm512_sub_pd({v[1]}_im, {v[5]}_im);")
        self.emit(f"  t2r = _mm512_add_pd({v[3]}_re, {v[7]}_re); t2i = _mm512_add_pd({v[3]}_im, {v[7]}_im);")
        self.emit(f"  t3r = _mm512_sub_pd({v[3]}_re, {v[7]}_re); t3i = _mm512_sub_pd({v[3]}_im, {v[7]}_im);")
        self.emit(f"  o0r = _mm512_add_pd(t0r, t2r); o0i = _mm512_add_pd(t0i, t2i);")
        self.emit(f"  o2r = _mm512_sub_pd(t0r, t2r); o2i = _mm512_sub_pd(t0i, t2i);")
        if fwd:
            self.emit(f"  o1r = _mm512_add_pd(t1r, t3i); o1i = _mm512_sub_pd(t1i, t3r);")
            self.emit(f"  o3r = _mm512_sub_pd(t1r, t3i); o3i = _mm512_add_pd(t1i, t3r);")
        else:
            self.emit(f"  o1r = _mm512_sub_pd(t1r, t3i); o1i = _mm512_add_pd(t1i, t3r);")
            self.emit(f"  o3r = _mm512_add_pd(t1r, t3i); o3i = _mm512_sub_pd(t1i, t3r);")
        
        # W₈ twiddle on odd
        if fwd:
            self.comment("W₈ [fwd]: o1*=W₈¹, o2*=-j, o3*=W₈³")
            self.emit(f"  t0r = _mm512_mul_pd(_mm512_add_pd(o1r, o1i), sqrt2_inv);")
            self.emit(f"  t0i = _mm512_mul_pd(_mm512_sub_pd(o1i, o1r), sqrt2_inv);")
            self.emit(f"  o1r = t0r; o1i = t0i;")
            self.emit(f"  t0r = o2i; t0i = _mm512_xor_pd(o2r, sign_flip);")
            self.emit(f"  o2r = t0r; o2i = t0i;")
            self.emit(f"  t0r = _mm512_mul_pd(_mm512_sub_pd(o3i, o3r), sqrt2_inv);")
            self.emit(f"  t0i = _mm512_xor_pd(_mm512_mul_pd(_mm512_add_pd(o3r, o3i), sqrt2_inv), sign_flip);")
            self.emit(f"  o3r = t0r; o3i = t0i;")
        else:
            self.comment("W₈ [bwd]: o1*=conj(W₈¹), o2*=+j, o3*=conj(W₈³)")
            self.emit(f"  t0r = _mm512_mul_pd(_mm512_sub_pd(o1r, o1i), sqrt2_inv);")
            self.emit(f"  t0i = _mm512_mul_pd(_mm512_add_pd(o1r, o1i), sqrt2_inv);")
            self.emit(f"  o1r = t0r; o1i = t0i;")
            self.emit(f"  t0r = _mm512_xor_pd(o2i, sign_flip); t0i = o2r;")
            self.emit(f"  o2r = t0r; o2i = t0i;")
            self.emit(f"  t0r = _mm512_xor_pd(_mm512_mul_pd(_mm512_add_pd(o3r, o3i), sqrt2_inv), sign_flip);")
            self.emit(f"  t0i = _mm512_mul_pd(_mm512_sub_pd(o3r, o3i), sqrt2_inv);")
            self.emit(f"  o3r = t0r; o3i = t0i;")
        
        # Combine
        self.comment("combine: y = even ± odd")
        self.emit(f"  {v[0]}_re = _mm512_add_pd(e0r, o0r); {v[0]}_im = _mm512_add_pd(e0i, o0i);")
        self.emit(f"  {v[4]}_re = _mm512_sub_pd(e0r, o0r); {v[4]}_im = _mm512_sub_pd(e0i, o0i);")
        self.emit(f"  {v[1]}_re = _mm512_add_pd(e1r, o1r); {v[1]}_im = _mm512_add_pd(e1i, o1i);")
        self.emit(f"  {v[5]}_re = _mm512_sub_pd(e1r, o1r); {v[5]}_im = _mm512_sub_pd(e1i, o1i);")
        self.emit(f"  {v[2]}_re = _mm512_add_pd(e2r, o2r); {v[2]}_im = _mm512_add_pd(e2i, o2i);")
        self.emit(f"  {v[6]}_re = _mm512_sub_pd(e2r, o2r); {v[6]}_im = _mm512_sub_pd(e2i, o2i);")
        self.emit(f"  {v[3]}_re = _mm512_add_pd(e3r, o3r); {v[3]}_im = _mm512_add_pd(e3i, o3i);")
        self.emit(f"  {v[7]}_re = _mm512_sub_pd(e3r, o3r); {v[7]}_im = _mm512_sub_pd(e3i, o3i);")
        self.emit(f"}}")
    
    # ──────────────────────────────────────────────────────────
    # Radix-16 butterfly (4×4 decomposition)
    # ──────────────────────────────────────────────────────────
    
    def emit_radix16_inline(self, v, direction, comment_str=""):
        """
        Inline radix-16 DIT on 16 variables v[0]..v[15] (in-place).
        
        4×4 decomposition (N₁=4, N₂=4):
          Sub-FFT j: radix-4 on {v[j], v[j+4], v[j+8], v[j+12]} for j=0..3
          Twiddle: W₁₆^{j·k₁} on sub-FFT j, output k₁
          Combine k₁: radix-4 on {v[k₁*4], v[k₁*4+1], v[k₁*4+2], v[k₁*4+3]}
        
        After sub-FFTs, variable layout:
          v[0]=Y₀[0], v[4]=Y₀[1], v[8]=Y₀[2],  v[12]=Y₀[3]
          v[1]=Y₁[0], v[5]=Y₁[1], v[9]=Y₁[2],  v[13]=Y₁[3]
          v[2]=Y₂[0], v[6]=Y₂[1], v[10]=Y₂[2], v[14]=Y₂[3]
          v[3]=Y₃[0], v[7]=Y₃[1], v[11]=Y₃[2], v[15]=Y₃[3]
        
        Combine groups (after twiddle):
          k₁=0: {v[0],v[1],v[2],v[3]}     → X[0],X[4],X[8],X[12]
          k₁=1: {v[4],v[5],v[6],v[7]}     → X[1],X[5],X[9],X[13]
          k₁=2: {v[8],v[9],v[10],v[11]}   → X[2],X[6],X[10],X[14]
          k₁=3: {v[12],v[13],v[14],v[15]} → X[3],X[7],X[11],X[15]
        """
        assert len(v) == 16
        
        if comment_str:
            self.comment(f"{comment_str} [{direction}]")
        
        # Pass 1: 4 radix-4 sub-FFTs
        for j in range(4):
            sub = [v[j], v[j+4], v[j+8], v[j+12]]
            self.comment(f"radix-16 sub-FFT j={j}: {sub[0]},{sub[1]},{sub[2]},{sub[3]}")
            self.emit_radix4_inline(sub, direction)
        
        # W₁₆ twiddle application
        # Y_j[k₁] is in v[j + 4*k₁]
        # Apply W₁₆^{j·k₁} for j=1..3, k₁=1..3
        self.comment("W₁₆ twiddles")
        for j in range(1, 4):
            for k1 in range(1, 4):
                e = (j * k1) % 16
                var_idx = j + 4 * k1
                self.emit_twiddle_16(v[var_idx], v[var_idx], e, direction)
        self.blank()
        
        # Pass 2: 4 radix-4 combines
        for k1 in range(4):
            sub = [v[k1*4], v[k1*4+1], v[k1*4+2], v[k1*4+3]]
            self.comment(f"radix-16 combine k₁={k1}: {sub[0]},{sub[1]},{sub[2]},{sub[3]}")
            self.emit_radix4_inline(sub, direction)
    
    # ──────────────────────────────────────────────────────────
    # Twiddle application
    # ──────────────────────────────────────────────────────────
    
    def emit_twiddle_16(self, dst, src, e16, direction):
        """Apply W₁₆^e16 (or conjugate for bwd) in-place. For radix-16 internal twiddles."""
        # Map W₁₆^e to our standard special-case checker
        is_special, typ = twiddle_is_trivial(e16, 16)
        self._emit_twiddle_impl(dst, src, e16, 16, is_special, typ, direction)
    
    def emit_twiddle_128(self, dst, src, e128, direction):
        """Apply W₁₂₈^e128 (or conjugate for bwd). For outer twiddles."""
        is_special, typ = twiddle_is_trivial(e128, 128)
        self._emit_twiddle_impl(dst, src, e128, 128, is_special, typ, direction)
    
    def _emit_twiddle_impl(self, dst, src, e, N, is_special, typ, direction):
        """Core twiddle emission — shared between W₁₆ and W₁₂₈."""
        fwd = (direction == 'fwd')
        
        if typ == 'one':
            if dst != src:
                self.emit(f"{dst}_re = {src}_re; {dst}_im = {src}_im;")
        elif typ == 'neg_one':
            self.emit(f"{dst}_re = _mm512_xor_pd({src}_re, sign_flip); {dst}_im = _mm512_xor_pd({src}_im, sign_flip);")
        elif typ == 'neg_j':
            if fwd:
                self.emit(f"{{ const __m512d t = {src}_re; {dst}_re = {src}_im; {dst}_im = _mm512_xor_pd(t, sign_flip); }}")
            else:
                self.emit(f"{{ const __m512d t = {src}_re; {dst}_re = _mm512_xor_pd({src}_im, sign_flip); {dst}_im = t; }}")
        elif typ == 'pos_j':
            if fwd:
                self.emit(f"{{ const __m512d t = {src}_re; {dst}_re = _mm512_xor_pd({src}_im, sign_flip); {dst}_im = t; }}")
            else:
                self.emit(f"{{ const __m512d t = {src}_re; {dst}_re = {src}_im; {dst}_im = _mm512_xor_pd(t, sign_flip); }}")
        elif typ == 'w8_1':
            if fwd:
                self.emit(f"{{ const __m512d tr = {src}_re, ti = {src}_im;")
                self.emit(f"  {dst}_re = _mm512_mul_pd(_mm512_add_pd(tr, ti), sqrt2_inv);")
                self.emit(f"  {dst}_im = _mm512_mul_pd(_mm512_sub_pd(ti, tr), sqrt2_inv); }}")
            else:
                self.emit(f"{{ const __m512d tr = {src}_re, ti = {src}_im;")
                self.emit(f"  {dst}_re = _mm512_mul_pd(_mm512_sub_pd(tr, ti), sqrt2_inv);")
                self.emit(f"  {dst}_im = _mm512_mul_pd(_mm512_add_pd(tr, ti), sqrt2_inv); }}")
        elif typ == 'w8_3':
            if fwd:
                self.emit(f"{{ const __m512d tr = {src}_re, ti = {src}_im;")
                self.emit(f"  {dst}_re = _mm512_mul_pd(_mm512_sub_pd(ti, tr), sqrt2_inv);")
                self.emit(f"  {dst}_im = _mm512_xor_pd(_mm512_mul_pd(_mm512_add_pd(tr, ti), sqrt2_inv), sign_flip); }}")
            else:
                self.emit(f"{{ const __m512d tr = {src}_re, ti = {src}_im;")
                self.emit(f"  {dst}_re = _mm512_xor_pd(_mm512_mul_pd(_mm512_add_pd(tr, ti), sqrt2_inv), sign_flip);")
                self.emit(f"  {dst}_im = _mm512_mul_pd(_mm512_sub_pd(tr, ti), sqrt2_inv); }}")
        elif typ == 'neg_w8_1':
            if fwd:
                self.emit(f"{{ const __m512d tr = {src}_re, ti = {src}_im;")
                self.emit(f"  {dst}_re = _mm512_xor_pd(_mm512_mul_pd(_mm512_add_pd(tr, ti), sqrt2_inv), sign_flip);")
                self.emit(f"  {dst}_im = _mm512_mul_pd(_mm512_sub_pd(tr, ti), sqrt2_inv); }}")
            else:
                self.emit(f"{{ const __m512d tr = {src}_re, ti = {src}_im;")
                self.emit(f"  {dst}_re = _mm512_mul_pd(_mm512_sub_pd(ti, tr), sqrt2_inv);")
                self.emit(f"  {dst}_im = _mm512_xor_pd(_mm512_mul_pd(_mm512_add_pd(tr, ti), sqrt2_inv), sign_flip); }}")
        elif typ == 'neg_w8_3':
            if fwd:
                self.emit(f"{{ const __m512d tr = {src}_re, ti = {src}_im;")
                self.emit(f"  {dst}_re = _mm512_mul_pd(_mm512_sub_pd(tr, ti), sqrt2_inv);")
                self.emit(f"  {dst}_im = _mm512_mul_pd(_mm512_add_pd(tr, ti), sqrt2_inv); }}")
            else:
                self.emit(f"{{ const __m512d tr = {src}_re, ti = {src}_im;")
                self.emit(f"  {dst}_re = _mm512_mul_pd(_mm512_add_pd(tr, ti), sqrt2_inv);")
                self.emit(f"  {dst}_im = _mm512_mul_pd(_mm512_sub_pd(ti, tr), sqrt2_inv); }}")
        else:
            # Generic cmul — use pre-hoisted twiddle ZMM
            key = (e % N, N)
            self.twiddles_needed.add(key)
            label = wN_label(e, N)
            if fwd:
                self.emit(f"{{ const __m512d tr = {src}_re;")
                self.emit(f"  {dst}_re = _mm512_fmsub_pd({src}_re, tw_{label}_re, _mm512_mul_pd({src}_im, tw_{label}_im));")
                self.emit(f"  {dst}_im = _mm512_fmadd_pd(tr, tw_{label}_im, _mm512_mul_pd({src}_im, tw_{label}_re)); }}")
            else:
                self.emit(f"{{ const __m512d tr = {src}_re;")
                self.emit(f"  {dst}_re = _mm512_fmadd_pd({src}_re, tw_{label}_re, _mm512_mul_pd({src}_im, tw_{label}_im));")
                self.emit(f"  {dst}_im = _mm512_fmsub_pd({src}_im, tw_{label}_re, _mm512_mul_pd(tr, tw_{label}_im)); }}")

# ──────────────────────────────────────────────────────────────────
# Collect all twiddle constants needed
# ──────────────────────────────────────────────────────────────────

def collect_twiddles():
    """Pre-scan: find all generic twiddle constants needed."""
    tw_set = set()
    
    # Inner W₁₆ twiddles (inside radix-16 sub-FFTs)
    for j in range(1, 4):
        for k1 in range(1, 4):
            e = (j * k1) % 16
            is_special, _ = twiddle_is_trivial(e, 16)
            if not is_special:
                tw_set.add((e, 16))
    
    # Outer W₁₂₈ twiddles (between passes)
    for n2 in range(1, 8):
        for k1 in range(1, 16):
            e = (n2 * k1) % 128
            is_special, _ = twiddle_is_trivial(e, 128)
            if not is_special:
                tw_set.add((e, 128))
    
    return tw_set

# ──────────────────────────────────────────────────────────────────
# Kernel generation
# ──────────────────────────────────────────────────────────────────

def emit_kernel(em, direction, tw_set):
    """Emit one kernel function (fwd or bwd)."""
    N = 128
    N1 = 16  # sub-FFT size
    N2 = 8   # number of sub-FFTs
    n_spill = N1 * N2  # 128
    
    dir_label = direction
    
    em.lines.append(f"static TARGET_AVX512 void")
    em.lines.append(f"radix128_n1_dit_kernel_{dir_label}_avx512(")
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
    em.emit(f"ALIGNAS_64 double spill_re[{n_spill} * 8];  /* {n_spill * 64} bytes */")
    em.emit(f"ALIGNAS_64 double spill_im[{n_spill} * 8];")
    em.blank()
    
    # Working registers: x0..x15 for pass 1 (radix-16), x0..x7 for pass 2 (radix-8)
    em.emit("__m512d x0_re, x0_im, x1_re, x1_im, x2_re, x2_im, x3_re, x3_im;")
    em.emit("__m512d x4_re, x4_im, x5_re, x5_im, x6_re, x6_im, x7_re, x7_im;")
    em.emit("__m512d x8_re, x8_im, x9_re, x9_im, x10_re, x10_im, x11_re, x11_im;")
    em.emit("__m512d x12_re, x12_im, x13_re, x13_im, x14_re, x14_im, x15_re, x15_im;")
    em.blank()
    
    # Hoisted twiddle broadcasts
    em.comment(f"Pre-broadcast twiddle constants [{dir_label}]")
    for (e, tN) in sorted(tw_set):
        label = wN_label(e, tN)
        em.emit(f"const __m512d tw_{label}_re = _mm512_set1_pd({label}_re);")
        em.emit(f"const __m512d tw_{label}_im = _mm512_set1_pd({label}_im);")
    em.blank()
    
    # k-loop
    em.emit("for (size_t k = 0; k < K; k += 8) {")
    em.indent += 1
    
    # ── PASS 1: 8 radix-16 sub-FFTs ──
    em.comment("═══════════════════════════════════════════════════")
    em.comment(f"PASS 1: {N2} radix-{N1} sub-FFTs → spill [{dir_label}]")
    em.comment("═══════════════════════════════════════════════════")
    em.blank()
    
    xvars16 = [f"x{i}" for i in range(16)]
    
    for n2 in range(N2):
        em.comment(f"── Sub-FFT n₂={n2} ──")
        # Load 16 inputs: x[N₂·n₁ + n₂] for n₁=0..15
        for n1 in range(N1):
            idx = N2 * n1 + n2
            em.emit_load_input(f"x{n1}", idx)
        em.blank()
        
        # Inline radix-16 butterfly
        em.emit_radix16_inline(xvars16, direction, f"radix-16 sub-FFT n₂={n2}")
        em.blank()
        
        # Spill 16 outputs
        # After radix-16 (4×4), the output layout in x variables:
        #   Combine k₁=0 produced: x0=X[0], x1=X[4], x2=X[8], x3=X[12]
        #   Combine k₁=1 produced: x4=X[1], x5=X[5], x6=X[9], x7=X[13]
        #   Combine k₁=2 produced: x8=X[2], x9=X[6], x10=X[10], x11=X[14]
        #   Combine k₁=3 produced: x12=X[3], x13=X[7], x14=X[11], x15=X[15]
        #
        # So Y_{n₂}[k₁] is at:
        #   k₁=0:  x0  (combine 0, output 0)
        #   k₁=1:  x4  (combine 1, output 0)
        #   k₁=2:  x8  (combine 2, output 0)
        #   k₁=3:  x12 (combine 3, output 0)
        #   k₁=4:  x1  (combine 0, output 1)
        #   k₁=5:  x5  (combine 1, output 1)
        #   k₁=6:  x9  (combine 2, output 1)
        #   k₁=7:  x13 (combine 3, output 1)
        #   k₁=8:  x2  (combine 0, output 2)
        #   k₁=9:  x6  (combine 1, output 2)
        #   k₁=10: x10 (combine 2, output 2)
        #   k₁=11: x14 (combine 3, output 2)
        #   k₁=12: x3  (combine 0, output 3)
        #   k₁=13: x7  (combine 1, output 3)
        #   k₁=14: x11 (combine 2, output 3)
        #   k₁=15: x15 (combine 3, output 3)
        #
        # General: Y_{n₂}[k₁] is at x[k₁_sub*4 + k₁_comb]
        # where k₁_comb = k₁ % 4, k₁_sub = k₁ // 4
        # → x_index = (k₁ % 4) * 4 + (k₁ // 4)  ... wait let me re-derive.
        #
        # The 4×4 inner structure:
        #   Sub-FFT j (j=0..3) operates on x[j], x[j+4], x[j+8], x[j+12]
        #   Combine k₁_inner (k₁_inner=0..3) operates on x[k₁_inner*4..k₁_inner*4+3]
        #   
        #   Combine k₁_inner produces outputs in order y[0..3] → stored back to
        #   x[k₁_inner*4], x[k₁_inner*4+1], x[k₁_inner*4+2], x[k₁_inner*4+3]
        #   These map to final DFT-16 output indices: k₁_inner + 4*k₂ for k₂=0..3
        #
        # So output X[k₁_inner + 4*k₂] is in x[k₁_inner*4 + k₂]
        # DFT-16 output index k₁ = k₁_inner + 4*k₂ 
        #   → k₁_inner = k₁ % 4, k₂ = k₁ // 4
        #   → x_index = (k₁ % 4) * 4 + (k₁ // 4)
        
        em.comment(f"spill sub-FFT {n2} (16 outputs)")
        for k1 in range(N1):
            k1_inner = k1 % 4
            k2 = k1 // 4
            x_idx = k1_inner * 4 + k2
            slot = n2 * N1 + k1
            em.emit_spill(f"x{x_idx}", slot)
        em.blank()
    
    # ── PASS 2: 16 radix-8 column combines ──
    em.comment("═══════════════════════════════════════════════════")
    em.comment(f"PASS 2: W₁₂₈ twiddles + {N1} radix-{N2} column combines [{dir_label}]")
    em.comment("═══════════════════════════════════════════════════")
    em.blank()
    
    xvars8 = [f"x{i}" for i in range(8)]
    
    for k1 in range(N1):
        em.comment(f"── Column k₁={k1} ──")
        
        # Reload 8 values: Y_{n₂}[k₁] for n₂=0..7
        for n2 in range(N2):
            slot = n2 * N1 + k1
            em.emit_reload(f"x{n2}", slot)
        em.blank()
        
        # Outer twiddles: W₁₂₈^{n₂·k₁}
        if k1 > 0:
            em.comment(f"W₁₂₈ twiddles for column k₁={k1} [{dir_label}]")
            for n2 in range(1, N2):
                e = (n2 * k1) % N
                em.emit_twiddle_128(f"x{n2}", f"x{n2}", e, direction)
            em.blank()
        
        # Radix-8 combine
        em.emit_radix8_inline(xvars8, direction, f"radix-8 combine column k₁={k1}")
        em.blank()
        
        # Store: X[k₁ + N₁·k₂] for k₂=0..7
        em.comment(f"store column {k1}")
        for k2 in range(N2):
            out_idx = k1 + N1 * k2
            em.emit_store_output(f"x{k2}", out_idx)
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
    em = Emitter(128)
    
    # Pre-collect all needed twiddles
    tw_set = collect_twiddles()
    
    # ── File header ──
    em.lines.append("/**")
    em.lines.append(" * @file fft_radix128_avx512_n1_gen.h")
    em.lines.append(" * @brief GENERATED DFT-128 AVX-512 N1 kernels (forward + backward)")
    em.lines.append(" *")
    em.lines.append(" * Auto-generated by gen_radix128_avx512.py.")
    em.lines.append(" * 16×8 Cooley-Tukey: 8 radix-16 sub-FFTs + 16 radix-8 combines.")
    em.lines.append(" * Radix-16 uses inline 4×4 decomposition.")
    em.lines.append(" * Forward and backward are native — no conjugate trick.")
    em.lines.append(" */")
    em.lines.append("")
    em.lines.append("#ifndef FFT_RADIX128_AVX512_N1_GEN_H")
    em.lines.append("#define FFT_RADIX128_AVX512_N1_GEN_H")
    em.lines.append("")
    em.lines.append("#include <immintrin.h>")
    em.lines.append("")
    
    # Twiddle scalar constants
    em.lines.append("/* ── Twiddle constants (guarded for multi-include) ── */")
    em.lines.append("#ifndef FFT_RADIX128_GEN_TWIDDLES_DEFINED")
    em.lines.append("#define FFT_RADIX128_GEN_TWIDDLES_DEFINED")
    for (e, tN) in sorted(tw_set):
        wr, wi = wN(e, tN)
        label = wN_label(e, tN)
        em.lines.append(f"static const double {label}_re = {wr:.20e};")
        em.lines.append(f"static const double {label}_im = {wi:.20e};")
    em.lines.append("#endif /* FFT_RADIX128_GEN_TWIDDLES_DEFINED */")
    em.lines.append("")
    
    # Load/store macros
    em.lines.append("#ifndef R128G_LD")
    em.lines.append("#define R128G_LD(p) _mm512_loadu_pd(p)")
    em.lines.append("#endif")
    em.lines.append("#ifndef R128G_ST")
    em.lines.append("#define R128G_ST(p,v) _mm512_storeu_pd((p),(v))")
    em.lines.append("#endif")
    em.lines.append("")
    
    # Forward kernel
    fwd_s, fwd_r = emit_kernel(em, 'fwd', tw_set)
    
    # Backward kernel
    bwd_s, bwd_r = emit_kernel(em, 'bwd', tw_set)
    
    em.lines.append("#endif /* FFT_RADIX128_AVX512_N1_GEN_H */")
    
    return em, tw_set, fwd_s, fwd_r, bwd_s, bwd_r

def main():
    em, tw_set, fwd_s, fwd_r, bwd_s, bwd_r = generate()
    
    print("\n".join(em.lines))
    
    # Count inner vs outer twiddles
    inner = sum(1 for e, N in tw_set if N == 16)
    outer = sum(1 for e, N in tw_set if N == 128)
    
    print(f"\n=== CODEGEN STATS (DFT-128) ===", file=sys.stderr)
    print(f"Forward:  {fwd_s} spills + {fwd_r} reloads = {fwd_s+fwd_r} L1 ops", file=sys.stderr)
    print(f"Backward: {bwd_s} spills + {bwd_r} reloads = {bwd_s+bwd_r} L1 ops", file=sys.stderr)
    print(f"Spill buffer: 128 slots = 16384 bytes (16 KB)", file=sys.stderr)
    print(f"Twiddle constants: {inner} inner (W₁₆) + {outer} outer (W₁₂₈) = {len(tw_set)} total", file=sys.stderr)
    print(f"Generated lines: {len(em.lines)}", file=sys.stderr)

if __name__ == '__main__':
    main()
