#!/usr/bin/env python3
"""
gen_fft_n1_avx512.py — Unified parametric FFT N1 codelet generator (AVX-512)

Generates straight-line DFT-N kernels using 2D Cooley-Tukey (N₁×N₂)
with inline sub-FFT butterflies and explicit spill management.

Supported sizes and their decompositions:
  N=16:  4×4   (4 radix-4 sub-FFTs,  4 radix-4 combines)  — 32 L1 ops
  N=32:  8×4   (4 radix-8 sub-FFTs,  8 radix-4 combines)  — 64 L1 ops
  N=64:  8×8   (8 radix-8 sub-FFTs,  8 radix-8 combines)  — 128 L1 ops
  N=128: 16×8  (8 radix-16 sub-FFTs, 16 radix-8 combines) — 256 L1 ops

Each size generates native forward AND backward kernels.
All twiddle broadcasts hoisted above the k-loop.

Usage:
  python3 gen_fft_n1_avx512.py 16  > fft_radix16_avx512_n1_gen.h
  python3 gen_fft_n1_avx512.py 32  > fft_radix32_avx512_n1_gen.h
  python3 gen_fft_n1_avx512.py 64  > fft_radix64_avx512_n1_gen.h
  python3 gen_fft_n1_avx512.py 128 > fft_radix128_avx512_n1_gen.h
  python3 gen_fft_n1_avx512.py all  # generates all sizes
"""

import math, sys

# ──────────────────────────────────────────────────────────────────
# Decomposition table
# ──────────────────────────────────────────────────────────────────

DECOMPOSITIONS = {
    16:  (4, 4),    # N₁=4,  N₂=4
    32:  (8, 4),    # N₁=8,  N₂=4
    64:  (8, 8),    # N₁=8,  N₂=8
    128: (16, 8),   # N₁=16, N₂=8
}

# ──────────────────────────────────────────────────────────────────
# Twiddle helpers
# ──────────────────────────────────────────────────────────────────

def wN(e, N):
    """W_N^e = cos(2πe/N) - j·sin(2πe/N)"""
    e = e % N
    angle = 2.0 * math.pi * e / N
    return (math.cos(angle), -math.sin(angle))

def wN_label(e, N):
    return f"W{N}_{e % N}"

def twiddle_is_trivial(e, N):
    """Check if W_N^e maps to an 8th root of unity."""
    e = e % N
    if e == 0:
        return True, 'one'
    if (8 * e) % N == 0:
        octant = (8 * e) // N
        types = ['one', 'w8_1', 'neg_j', 'w8_3',
                 'neg_one', 'neg_w8_1', 'pos_j', 'neg_w8_3']
        return True, types[octant % 8]
    return False, 'cmul'

# ──────────────────────────────────────────────────────────────────
# Emitter
# ──────────────────────────────────────────────────────────────────

class Emitter:
    def __init__(self, N):
        self.N = N
        self.lines = []
        self.indent = 1
        self.spill_count = 0
        self.reload_count = 0
        self.twiddles_needed = set()
        self.macro_prefix = f"R{N}G"
    
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
        self.emit(f"{var}_re = {self.macro_prefix}_LD(&in_re[{n} * K + k]);")
        self.emit(f"{var}_im = {self.macro_prefix}_LD(&in_im[{n} * K + k]);")
    
    def emit_store_output(self, var, m):
        self.emit(f"{self.macro_prefix}_ST(&out_re[{m} * K + k], {var}_re);")
        self.emit(f"{self.macro_prefix}_ST(&out_im[{m} * K + k], {var}_im);")
    
    # ──────────────────────────────────────────────────────────
    # Radix-4 inline butterfly
    # ──────────────────────────────────────────────────────────
    
    def emit_radix4(self, v, direction):
        """Radix-4 DIT on 4 vars v[0..3], in-place."""
        fwd = (direction == 'fwd')
        a, b, c, d = v[0], v[1], v[2], v[3]
        self.emit(f"{{ __m512d t0r, t0i, t1r, t1i, t2r, t2i, t3r, t3i;")
        self.emit(f"  t0r = _mm512_add_pd({a}_re, {c}_re); t0i = _mm512_add_pd({a}_im, {c}_im);")
        self.emit(f"  t1r = _mm512_sub_pd({a}_re, {c}_re); t1i = _mm512_sub_pd({a}_im, {c}_im);")
        self.emit(f"  t2r = _mm512_add_pd({b}_re, {d}_re); t2i = _mm512_add_pd({b}_im, {d}_im);")
        self.emit(f"  t3r = _mm512_sub_pd({b}_re, {d}_re); t3i = _mm512_sub_pd({b}_im, {d}_im);")
        self.emit(f"  {a}_re = _mm512_add_pd(t0r, t2r); {a}_im = _mm512_add_pd(t0i, t2i);")
        self.emit(f"  {c}_re = _mm512_sub_pd(t0r, t2r); {c}_im = _mm512_sub_pd(t0i, t2i);")
        if fwd:
            self.emit(f"  {b}_re = _mm512_add_pd(t1r, t3i); {b}_im = _mm512_sub_pd(t1i, t3r);")
            self.emit(f"  {d}_re = _mm512_sub_pd(t1r, t3i); {d}_im = _mm512_add_pd(t1i, t3r);")
        else:
            self.emit(f"  {b}_re = _mm512_sub_pd(t1r, t3i); {b}_im = _mm512_add_pd(t1i, t3r);")
            self.emit(f"  {d}_re = _mm512_add_pd(t1r, t3i); {d}_im = _mm512_sub_pd(t1i, t3r);")
        self.emit(f"}}")
    
    # ──────────────────────────────────────────────────────────
    # Radix-8 inline butterfly
    # ──────────────────────────────────────────────────────────
    
    def emit_radix8(self, v, direction, comment_str=""):
        """Radix-8 DIT on 8 vars v[0..7], in-place. Split-radix: 2×radix-4 + W₈ + combine."""
        assert len(v) == 8
        fwd = (direction == 'fwd')
        if comment_str:
            self.comment(f"{comment_str} [{direction}]")
        
        self.emit(f"{{ __m512d e0r, e0i, e1r, e1i, e2r, e2i, e3r, e3i;")
        self.emit(f"  __m512d t0r, t0i, t1r, t1i, t2r, t2i, t3r, t3i;")
        # Even radix-4: v[0], v[2], v[4], v[6]
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
        
        # W₈ twiddle
        if fwd:
            self.emit(f"  t0r = _mm512_mul_pd(_mm512_add_pd(o1r, o1i), sqrt2_inv);")
            self.emit(f"  t0i = _mm512_mul_pd(_mm512_sub_pd(o1i, o1r), sqrt2_inv);")
            self.emit(f"  o1r = t0r; o1i = t0i;")
            self.emit(f"  t0r = o2i; t0i = _mm512_xor_pd(o2r, sign_flip);")
            self.emit(f"  o2r = t0r; o2i = t0i;")
            self.emit(f"  t0r = _mm512_mul_pd(_mm512_sub_pd(o3i, o3r), sqrt2_inv);")
            self.emit(f"  t0i = _mm512_xor_pd(_mm512_mul_pd(_mm512_add_pd(o3r, o3i), sqrt2_inv), sign_flip);")
            self.emit(f"  o3r = t0r; o3i = t0i;")
        else:
            self.emit(f"  t0r = _mm512_mul_pd(_mm512_sub_pd(o1r, o1i), sqrt2_inv);")
            self.emit(f"  t0i = _mm512_mul_pd(_mm512_add_pd(o1r, o1i), sqrt2_inv);")
            self.emit(f"  o1r = t0r; o1i = t0i;")
            self.emit(f"  t0r = _mm512_xor_pd(o2i, sign_flip); t0i = o2r;")
            self.emit(f"  o2r = t0r; o2i = t0i;")
            self.emit(f"  t0r = _mm512_xor_pd(_mm512_mul_pd(_mm512_add_pd(o3r, o3i), sqrt2_inv), sign_flip);")
            self.emit(f"  t0i = _mm512_mul_pd(_mm512_sub_pd(o3r, o3i), sqrt2_inv);")
            self.emit(f"  o3r = t0r; o3i = t0i;")
        
        # Combine
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
    # Radix-16 inline butterfly (4×4)
    # ──────────────────────────────────────────────────────────
    
    def emit_radix16(self, v, direction, comment_str=""):
        """Radix-16 DIT on 16 vars, in-place. 4×4: four radix-4 + W₁₆ twiddle + four radix-4."""
        assert len(v) == 16
        if comment_str:
            self.comment(f"{comment_str} [{direction}]")
        
        # Pass 1: 4 radix-4 sub-FFTs
        for j in range(4):
            sub = [v[j], v[j+4], v[j+8], v[j+12]]
            self.emit_radix4(sub, direction)
        
        # W₁₆ twiddles: Y_j[k₁] at v[j + 4*k₁], apply W₁₆^{j·k₁}
        for j in range(1, 4):
            for k1 in range(1, 4):
                e = (j * k1) % 16
                self.emit_twiddle(v[j + 4*k1], v[j + 4*k1], e, 16, direction)
        
        # Pass 2: 4 radix-4 combines
        for k1 in range(4):
            sub = [v[k1*4], v[k1*4+1], v[k1*4+2], v[k1*4+3]]
            self.emit_radix4(sub, direction)
    
    # ──────────────────────────────────────────────────────────
    # Generic inline butterfly dispatcher
    # ──────────────────────────────────────────────────────────
    
    def emit_butterfly(self, size, v, direction, comment_str=""):
        """Dispatch to the right inline butterfly by size."""
        if size == 4:
            if comment_str:
                self.comment(f"{comment_str} [{direction}]")
            self.emit_radix4(v, direction)
        elif size == 8:
            self.emit_radix8(v, direction, comment_str)
        elif size == 16:
            self.emit_radix16(v, direction, comment_str)
        else:
            raise ValueError(f"No inline butterfly for size {size}")
    
    # ──────────────────────────────────────────────────────────
    # Output index mapping for N₁×N₂ decomposition
    # ──────────────────────────────────────────────────────────
    
    def subfft_output_var_index(self, k1, N1):
        """
        After an inline radix-N₁ butterfly on x0..x{N1-1},
        which variable holds DFT output index k₁?
        
        For radix-4: trivial, output k maps to x[k] (in-place indexing)
          k=0→x0, k=1→x1, k=2→x2, k=3→x3
          
        For radix-8: even/odd split maps output k to x[k] directly.
          k=0→x0, k=1→x1, ..., k=7→x7
          
        For radix-16 (4×4): output X[k₁_inner + 4*k₂]
          is in x[k₁_inner*4 + k₂] where k₁_inner = k₁%4, k₂ = k₁//4
        """
        if N1 <= 8:
            return k1
        elif N1 == 16:
            return (k1 % 4) * 4 + (k1 // 4)
        else:
            raise ValueError(f"Unknown output mapping for N1={N1}")
    
    # ──────────────────────────────────────────────────────────
    # Twiddle emission
    # ──────────────────────────────────────────────────────────
    
    def emit_twiddle(self, dst, src, e, tN, direction):
        """Apply W_tN^e (or conjugate for bwd)."""
        is_special, typ = twiddle_is_trivial(e, tN)
        fwd = (direction == 'fwd')
        
        if typ == 'one':
            if dst != src:
                self.emit(f"{dst}_re = {src}_re; {dst}_im = {src}_im;")
        elif typ == 'neg_one':
            self.emit(f"{dst}_re = _mm512_xor_pd({src}_re, sign_flip); "
                      f"{dst}_im = _mm512_xor_pd({src}_im, sign_flip);")
        elif typ == 'neg_j':
            if fwd:
                self.emit(f"{{ const __m512d t = {src}_re; {dst}_re = {src}_im; "
                          f"{dst}_im = _mm512_xor_pd(t, sign_flip); }}")
            else:
                self.emit(f"{{ const __m512d t = {src}_re; "
                          f"{dst}_re = _mm512_xor_pd({src}_im, sign_flip); {dst}_im = t; }}")
        elif typ == 'pos_j':
            if fwd:
                self.emit(f"{{ const __m512d t = {src}_re; "
                          f"{dst}_re = _mm512_xor_pd({src}_im, sign_flip); {dst}_im = t; }}")
            else:
                self.emit(f"{{ const __m512d t = {src}_re; {dst}_re = {src}_im; "
                          f"{dst}_im = _mm512_xor_pd(t, sign_flip); }}")
        elif typ == 'w8_1':
            self.emit(f"{{ const __m512d tr = {src}_re, ti = {src}_im;")
            if fwd:
                self.emit(f"  {dst}_re = _mm512_mul_pd(_mm512_add_pd(tr, ti), sqrt2_inv);")
                self.emit(f"  {dst}_im = _mm512_mul_pd(_mm512_sub_pd(ti, tr), sqrt2_inv); }}")
            else:
                self.emit(f"  {dst}_re = _mm512_mul_pd(_mm512_sub_pd(tr, ti), sqrt2_inv);")
                self.emit(f"  {dst}_im = _mm512_mul_pd(_mm512_add_pd(tr, ti), sqrt2_inv); }}")
        elif typ == 'w8_3':
            self.emit(f"{{ const __m512d tr = {src}_re, ti = {src}_im;")
            if fwd:
                self.emit(f"  {dst}_re = _mm512_mul_pd(_mm512_sub_pd(ti, tr), sqrt2_inv);")
                self.emit(f"  {dst}_im = _mm512_xor_pd(_mm512_mul_pd(_mm512_add_pd(tr, ti), sqrt2_inv), sign_flip); }}")
            else:
                self.emit(f"  {dst}_re = _mm512_xor_pd(_mm512_mul_pd(_mm512_add_pd(tr, ti), sqrt2_inv), sign_flip);")
                self.emit(f"  {dst}_im = _mm512_mul_pd(_mm512_sub_pd(tr, ti), sqrt2_inv); }}")
        elif typ == 'neg_w8_1':
            self.emit(f"{{ const __m512d tr = {src}_re, ti = {src}_im;")
            if fwd:
                self.emit(f"  {dst}_re = _mm512_xor_pd(_mm512_mul_pd(_mm512_add_pd(tr, ti), sqrt2_inv), sign_flip);")
                self.emit(f"  {dst}_im = _mm512_mul_pd(_mm512_sub_pd(tr, ti), sqrt2_inv); }}")
            else:
                self.emit(f"  {dst}_re = _mm512_mul_pd(_mm512_sub_pd(ti, tr), sqrt2_inv);")
                self.emit(f"  {dst}_im = _mm512_xor_pd(_mm512_mul_pd(_mm512_add_pd(tr, ti), sqrt2_inv), sign_flip); }}")
        elif typ == 'neg_w8_3':
            self.emit(f"{{ const __m512d tr = {src}_re, ti = {src}_im;")
            if fwd:
                self.emit(f"  {dst}_re = _mm512_mul_pd(_mm512_sub_pd(tr, ti), sqrt2_inv);")
                self.emit(f"  {dst}_im = _mm512_mul_pd(_mm512_add_pd(tr, ti), sqrt2_inv); }}")
            else:
                self.emit(f"  {dst}_re = _mm512_mul_pd(_mm512_add_pd(tr, ti), sqrt2_inv);")
                self.emit(f"  {dst}_im = _mm512_mul_pd(_mm512_sub_pd(ti, tr), sqrt2_inv); }}")
        else:
            # Generic cmul with hoisted twiddle
            key = (e % tN, tN)
            self.twiddles_needed.add(key)
            label = wN_label(e, tN)
            self.emit(f"{{ const __m512d tr = {src}_re;")
            if fwd:
                self.emit(f"  {dst}_re = _mm512_fmsub_pd({src}_re, tw_{label}_re, _mm512_mul_pd({src}_im, tw_{label}_im));")
                self.emit(f"  {dst}_im = _mm512_fmadd_pd(tr, tw_{label}_im, _mm512_mul_pd({src}_im, tw_{label}_re)); }}")
            else:
                self.emit(f"  {dst}_re = _mm512_fmadd_pd({src}_re, tw_{label}_re, _mm512_mul_pd({src}_im, tw_{label}_im));")
                self.emit(f"  {dst}_im = _mm512_fmsub_pd({src}_im, tw_{label}_re, _mm512_mul_pd(tr, tw_{label}_im)); }}")

# ──────────────────────────────────────────────────────────────────
# Twiddle collection (pre-scan)
# ──────────────────────────────────────────────────────────────────

def collect_twiddles(N, N1, N2):
    """Find all generic twiddle constants needed for this decomposition."""
    tw_set = set()
    
    # Inner twiddles (inside radix-N1 sub-FFTs, if N1=16)
    if N1 == 16:
        for j in range(1, 4):
            for k1 in range(1, 4):
                e = (j * k1) % 16
                is_special, _ = twiddle_is_trivial(e, 16)
                if not is_special:
                    tw_set.add((e, 16))
    
    # Outer twiddles: W_N^{n2·k1} for n2=1..N2-1, k1=1..N1-1
    for n2 in range(1, N2):
        for k1 in range(1, N1):
            e = (n2 * k1) % N
            is_special, _ = twiddle_is_trivial(e, N)
            if not is_special:
                tw_set.add((e, N))
    
    return tw_set

# ──────────────────────────────────────────────────────────────────
# Kernel generation
# ──────────────────────────────────────────────────────────────────

def emit_kernel(em, direction, N, N1, N2, tw_set):
    """Emit one kernel function (fwd or bwd) for DFT-N using N1×N2."""
    dir_label = direction
    n_spill = N1 * N2
    
    em.lines.append(f"static TARGET_AVX512 void")
    em.lines.append(f"radix{N}_n1_dit_kernel_{dir_label}_avx512(")
    em.lines.append(f"    const double *RESTRICT in_re, const double *RESTRICT in_im,")
    em.lines.append(f"    double *RESTRICT out_re, double *RESTRICT out_im,")
    em.lines.append(f"    size_t K)")
    em.lines.append(f"{{")
    
    em.indent = 1
    em.spill_count = 0
    em.reload_count = 0
    
    em.emit("const __m512d sign_flip = _mm512_set1_pd(-0.0);")
    em.emit("const __m512d sqrt2_inv = _mm512_set1_pd(0.70710678118654752440);")
    em.blank()
    
    em.emit(f"ALIGNAS_64 double spill_re[{n_spill} * 8];")
    em.emit(f"ALIGNAS_64 double spill_im[{n_spill} * 8];")
    em.blank()
    
    # Declare enough working registers
    max_vars = max(N1, N2)
    for i in range(0, max_vars, 4):
        chunk = min(4, max_vars - i)
        parts = [f"x{i+j}_re, x{i+j}_im" for j in range(chunk)]
        em.emit(f"__m512d {', '.join(parts)};")
    em.blank()
    
    # Hoisted twiddle broadcasts
    if tw_set:
        em.comment(f"Pre-broadcast twiddle constants [{dir_label}]")
        for (e, tN) in sorted(tw_set):
            label = wN_label(e, tN)
            em.emit(f"const __m512d tw_{label}_re = _mm512_set1_pd({label}_re);")
            em.emit(f"const __m512d tw_{label}_im = _mm512_set1_pd({label}_im);")
        em.blank()
    
    # k-loop
    em.emit("for (size_t k = 0; k < K; k += 8) {")
    em.indent += 1
    
    # ── PASS 1: N₂ sub-FFTs of size N₁ ──
    em.comment(f"PASS 1: {N2} radix-{N1} sub-FFTs [{dir_label}]")
    em.blank()
    
    xvars_p1 = [f"x{i}" for i in range(N1)]
    
    for n2 in range(N2):
        em.comment(f"sub-FFT n₂={n2}")
        # Load N₁ inputs: x[N₂·n₁ + n₂]
        for n1 in range(N1):
            em.emit_load_input(f"x{n1}", N2 * n1 + n2)
        em.blank()
        
        em.emit_butterfly(N1, xvars_p1, direction, f"radix-{N1} sub-FFT n₂={n2}")
        em.blank()
        
        # Spill N₁ outputs
        for k1 in range(N1):
            x_idx = em.subfft_output_var_index(k1, N1)
            em.emit_spill(f"x{x_idx}", n2 * N1 + k1)
        em.blank()
    
    # ── PASS 2: N₁ column combines of size N₂ ──
    em.comment(f"PASS 2: {N1} radix-{N2} combines with W_{N} twiddles [{dir_label}]")
    em.blank()
    
    xvars_p2 = [f"x{i}" for i in range(N2)]
    
    for k1 in range(N1):
        em.comment(f"column k₁={k1}")
        for n2 in range(N2):
            em.emit_reload(f"x{n2}", n2 * N1 + k1)
        em.blank()
        
        if k1 > 0:
            for n2 in range(1, N2):
                e = (n2 * k1) % N
                em.emit_twiddle(f"x{n2}", f"x{n2}", e, N, direction)
            em.blank()
        
        em.emit_butterfly(N2, xvars_p2, direction, f"radix-{N2} combine k₁={k1}")
        em.blank()
        
        for k2 in range(N2):
            em.emit_store_output(f"x{k2}", k1 + N1 * k2)
        em.blank()
    
    em.indent -= 1
    em.emit("} /* end k-loop */")
    em.lines.append("}")
    em.lines.append("")
    
    return em.spill_count, em.reload_count

# ──────────────────────────────────────────────────────────────────
# File generation
# ──────────────────────────────────────────────────────────────────

def generate(N):
    N1, N2 = DECOMPOSITIONS[N]
    em = Emitter(N)
    tw_set = collect_twiddles(N, N1, N2)
    mp = em.macro_prefix
    
    em.lines.append(f"/**")
    em.lines.append(f" * @file fft_radix{N}_avx512_n1_gen.h")
    em.lines.append(f" * @brief GENERATED DFT-{N} AVX-512 N1 kernels (fwd + bwd)")
    em.lines.append(f" *")
    em.lines.append(f" * {N1}×{N2} Cooley-Tukey: {N2} radix-{N1} sub-FFTs + {N1} radix-{N2} combines.")
    em.lines.append(f" * Spill: {N1*N2} stores + {N1*N2} loads = {2*N1*N2} L1 ops")
    em.lines.append(f" * Generated by gen_fft_n1_avx512.py")
    em.lines.append(f" */")
    em.lines.append(f"")
    em.lines.append(f"#ifndef FFT_RADIX{N}_AVX512_N1_GEN_H")
    em.lines.append(f"#define FFT_RADIX{N}_AVX512_N1_GEN_H")
    em.lines.append(f"")
    em.lines.append(f"#include <immintrin.h>")
    em.lines.append(f"")
    
    # Twiddle constants — grouped by twiddle-N to allow sharing across files
    if tw_set:
        # Group by twiddle size
        by_tN = {}
        for (e, tN) in sorted(tw_set):
            by_tN.setdefault(tN, []).append(e)
        for tN in sorted(by_tN):
            guard = f"FFT_W{tN}_TWIDDLES_DEFINED"
            em.lines.append(f"#ifndef {guard}")
            em.lines.append(f"#define {guard}")
            for e in sorted(by_tN[tN]):
                wr, wi = wN(e, tN)
                label = wN_label(e, tN)
                em.lines.append(f"static const double {label}_re = {wr:.20e};")
                em.lines.append(f"static const double {label}_im = {wi:.20e};")
            em.lines.append(f"#endif /* {guard} */")
            em.lines.append(f"")
    
    # Load/store macros
    em.lines.append(f"#ifndef {mp}_LD")
    em.lines.append(f"#define {mp}_LD(p) _mm512_loadu_pd(p)")
    em.lines.append(f"#endif")
    em.lines.append(f"#ifndef {mp}_ST")
    em.lines.append(f"#define {mp}_ST(p,v) _mm512_storeu_pd((p),(v))")
    em.lines.append(f"#endif")
    em.lines.append(f"")
    
    fwd_s, fwd_r = emit_kernel(em, 'fwd', N, N1, N2, tw_set)
    bwd_s, bwd_r = emit_kernel(em, 'bwd', N, N1, N2, tw_set)
    
    em.lines.append(f"#endif /* FFT_RADIX{N}_AVX512_N1_GEN_H */")
    
    return em, tw_set, N1, N2, fwd_s, fwd_r, bwd_s, bwd_r

def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <N|all>", file=sys.stderr)
        print(f"  N ∈ {{{', '.join(str(k) for k in sorted(DECOMPOSITIONS))}}}", file=sys.stderr)
        sys.exit(1)
    
    if sys.argv[1] == 'all':
        sizes = sorted(DECOMPOSITIONS.keys())
    else:
        sizes = [int(sys.argv[1])]
    
    for N in sizes:
        if N not in DECOMPOSITIONS:
            print(f"Error: N={N} not supported. Use {sorted(DECOMPOSITIONS.keys())}", file=sys.stderr)
            sys.exit(1)
        
        em, tw_set, N1, N2, fs, fr, bs, br = generate(N)
        
        if len(sizes) == 1:
            # Single size: output to stdout
            print("\n".join(em.lines))
        else:
            # Multiple sizes: write to files
            fname = f"fft_radix{N}_avx512_n1_gen.h"
            with open(fname, 'w') as f:
                f.write("\n".join(em.lines))
            print(f"  Wrote {fname}", file=sys.stderr)
        
        inner = sum(1 for e, tN in tw_set if tN != N)
        outer = sum(1 for e, tN in tw_set if tN == N)
        total_spill = fs + fr
        
        print(f"\n=== DFT-{N} ({N1}×{N2}) ===", file=sys.stderr)
        print(f"  Spill:    {fs}+{fr} = {total_spill} L1 ops", file=sys.stderr)
        print(f"  Twiddles: {inner} inner + {outer} outer = {len(tw_set)} generic", file=sys.stderr)
        print(f"  Lines:    {len(em.lines)}", file=sys.stderr)

if __name__ == '__main__':
    main()
