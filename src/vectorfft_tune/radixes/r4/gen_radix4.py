#!/usr/bin/env python3
#
# Copyright (c) 2025 Tugbars Heptaskin
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
"""
gen_radix4.py — DFT-4 standalone codelets for VectorFFT

DFT-4 = 2×DFT-2 + ×(-j) combine. 16 adds, 0 muls, 0 constants.
Forward/backward differ only in the ×j sign on the odd-diff term.

Twiddle table: 3*K doubles per component (W^1, W^2, W^3).

Usage: python3 gen_radix4.py avx512|avx2|scalar
"""
import sys

ISA = {
    'avx512': {'VL':8,'V':'__m512d','p':'_mm512',
        'target':'__attribute__((target("avx512f,fma")))',
        'guard_begin':'#if defined(__AVX512F__) || defined(__AVX512F)',
        'guard_end':'#endif /* __AVX512F__ */'},
    'avx2': {'VL':4,'V':'__m256d','p':'_mm256',
        'target':'__attribute__((target("avx2,fma")))',
        'guard_begin':'#ifdef __AVX2__',
        'guard_end':'#endif /* __AVX2__ */'},
    'scalar': {'VL':1,'V':'double','p':'',
        'target':'',
        'guard_begin':'',
        'guard_end':''},
}

def gen_scalar_notw(direction):
    # Forward: X[1].re = t1_re + t3_im,  X[1].im = t1_im - t3_re
    #          X[3].re = t1_re - t3_im,  X[3].im = t1_im + t3_re
    # Backward: swap X[1] and X[3] roles (j → -j)
    fwd = direction == 'fwd'
    if fwd:
        r1 = 't1r + t3i'; i1 = 't1i - t3r'
        r3 = 't1r - t3i'; i3 = 't1i + t3r'
    else:
        r1 = 't1r - t3i'; i1 = 't1i + t3r'
        r3 = 't1r + t3i'; i3 = 't1i - t3r'
    return f"""    for (size_t k = 0; k < K; k++) {{
        /* Even: x[0] ± x[2] */
        const double t0r = in_re[0*K+k] + in_re[2*K+k];
        const double t0i = in_im[0*K+k] + in_im[2*K+k];
        const double t1r = in_re[0*K+k] - in_re[2*K+k];
        const double t1i = in_im[0*K+k] - in_im[2*K+k];
        /* Odd: x[1] ± x[3] */
        const double t2r = in_re[1*K+k] + in_re[3*K+k];
        const double t2i = in_im[1*K+k] + in_im[3*K+k];
        const double t3r = in_re[1*K+k] - in_re[3*K+k];
        const double t3i = in_im[1*K+k] - in_im[3*K+k];
        /* Combine */
        out_re[0*K+k] = t0r + t2r;
        out_im[0*K+k] = t0i + t2i;
        out_re[2*K+k] = t0r - t2r;
        out_im[2*K+k] = t0i - t2i;
        out_re[1*K+k] = {r1};
        out_im[1*K+k] = {i1};
        out_re[3*K+k] = {r3};
        out_im[3*K+k] = {i3};
    }}"""

def gen_scalar_tw(direction):
    fwd = direction == 'fwd'
    # cmul fwd: xr*wr - xi*wi, xr*wi + xi*wr
    # cmul bwd: xr*wr + xi*wi, -xr*wi + xi*wr
    if fwd:
        cm = lambda vr,vi,wr,wi: (f'{vr}*{wr} - {vi}*{wi}', f'{vr}*{wi} + {vi}*{wr}')
        r1 = 't1r + t3i'; i1 = 't1i - t3r'
        r3 = 't1r - t3i'; i3 = 't1i + t3r'
    else:
        cm = lambda vr,vi,wr,wi: (f'{vr}*{wr} + {vi}*{wi}', f'-{vr}*{wi} + {vi}*{wr}')
        r1 = 't1r - t3i'; i1 = 't1i + t3r'
        r3 = 't1r + t3i'; i3 = 't1i - t3r'
    c1r, c1i = cm('r1r','r1i','w1r','w1i')
    c2r, c2i = cm('r2r','r2i','w2r','w2i')
    c3r, c3i = cm('r3r','r3i','w3r','w3i')
    return f"""    for (size_t k = 0; k < K; k++) {{
        const double x0r = in_re[0*K+k], x0i = in_im[0*K+k];
        const double r1r = in_re[1*K+k], r1i = in_im[1*K+k];
        const double r2r = in_re[2*K+k], r2i = in_im[2*K+k];
        const double r3r = in_re[3*K+k], r3i = in_im[3*K+k];
        const double w1r = tw_re[0*K+k], w1i = tw_im[0*K+k];
        const double w2r = tw_re[1*K+k], w2i = tw_im[1*K+k];
        const double w3r = tw_re[2*K+k], w3i = tw_im[2*K+k];
        const double x1r = {c1r}, x1i = {c1i};
        const double x2r = {c2r}, x2i = {c2i};
        const double x3r = {c3r}, x3i = {c3i};
        const double t0r = x0r + x2r, t0i = x0i + x2i;
        const double t1r = x0r - x2r, t1i = x0i - x2i;
        const double t2r = x1r + x3r, t2i = x1i + x3i;
        const double t3r = x1r - x3r, t3i = x1i - x3i;
        out_re[0*K+k] = t0r + t2r; out_im[0*K+k] = t0i + t2i;
        out_re[2*K+k] = t0r - t2r; out_im[2*K+k] = t0i - t2i;
        out_re[1*K+k] = {r1}; out_im[1*K+k] = {i1};
        out_re[3*K+k] = {r3}; out_im[3*K+k] = {i3};
    }}"""

def gen_simd_notw(isa_name, direction):
    I = ISA[isa_name]
    V, p, VL = I['V'], I['p'], I['VL']
    ADD, SUB = f'{p}_add_pd', f'{p}_sub_pd'
    fwd = direction == 'fwd'
    # Forward: X[1].re = t1r + t3i,  X[1].im = t1i - t3r
    # Backward: swap
    if fwd:
        r1, i1 = f'{ADD}(t1r, t3i)', f'{SUB}(t1i, t3r)'
        r3, i3 = f'{SUB}(t1r, t3i)', f'{ADD}(t1i, t3r)'
    else:
        r1, i1 = f'{SUB}(t1r, t3i)', f'{ADD}(t1i, t3r)'
        r3, i3 = f'{ADD}(t1r, t3i)', f'{SUB}(t1i, t3r)'

    return f"""    for (size_t k = 0; k < K; k += {VL}) {{
        const {V} x0r = R4_LD(&in_re[0*K+k]), x0i = R4_LD(&in_im[0*K+k]);
        const {V} x1r = R4_LD(&in_re[1*K+k]), x1i = R4_LD(&in_im[1*K+k]);
        const {V} x2r = R4_LD(&in_re[2*K+k]), x2i = R4_LD(&in_im[2*K+k]);
        const {V} x3r = R4_LD(&in_re[3*K+k]), x3i = R4_LD(&in_im[3*K+k]);
        const {V} t0r = {ADD}(x0r, x2r), t0i = {ADD}(x0i, x2i);
        const {V} t1r = {SUB}(x0r, x2r), t1i = {SUB}(x0i, x2i);
        const {V} t2r = {ADD}(x1r, x3r), t2i = {ADD}(x1i, x3i);
        const {V} t3r = {SUB}(x1r, x3r), t3i = {SUB}(x1i, x3i);
        R4_ST(&out_re[0*K+k], {ADD}(t0r, t2r)); R4_ST(&out_im[0*K+k], {ADD}(t0i, t2i));
        R4_ST(&out_re[2*K+k], {SUB}(t0r, t2r)); R4_ST(&out_im[2*K+k], {SUB}(t0i, t2i));
        R4_ST(&out_re[1*K+k], {r1}); R4_ST(&out_im[1*K+k], {i1});
        R4_ST(&out_re[3*K+k], {r3}); R4_ST(&out_im[3*K+k], {i3});
    }}"""

def gen_simd_tw(isa_name, direction):
    I = ISA[isa_name]
    V, p, VL = I['V'], I['p'], I['VL']
    ADD, SUB = f'{p}_add_pd', f'{p}_sub_pd'
    MUL = f'{p}_mul_pd'
    FNMA = f'{p}_fnmadd_pd'  # -a*b + c
    FMA  = f'{p}_fmadd_pd'   #  a*b + c
    fwd = direction == 'fwd'
    # fwd cmul: re = fnmadd(xi, wi, mul(xr, wr)), im = fmadd(xr, wi, mul(xi, wr))
    # bwd cmul: re = fmadd(xi, wi, mul(xr, wr)),  im = fnmadd(xr, wi, mul(xi, wr))
    if fwd:
        def cmul(xr,xi,wr,wi): return (f'{FNMA}({xi},{wi},{MUL}({xr},{wr}))',
                                        f'{FMA}({xr},{wi},{MUL}({xi},{wr}))')
        r1, i1 = lambda: f'{ADD}(t1r, t3i)', lambda: f'{SUB}(t1i, t3r)'
        r3, i3 = lambda: f'{SUB}(t1r, t3i)', lambda: f'{ADD}(t1i, t3r)'
    else:
        def cmul(xr,xi,wr,wi): return (f'{FMA}({xi},{wi},{MUL}({xr},{wr}))',
                                        f'{FNMA}({xr},{wi},{MUL}({xi},{wr}))')
        r1, i1 = lambda: f'{SUB}(t1r, t3i)', lambda: f'{ADD}(t1i, t3r)'
        r3, i3 = lambda: f'{ADD}(t1r, t3i)', lambda: f'{SUB}(t1i, t3r)'

    c1r, c1i = cmul('r1r','r1i','w1r','w1i')
    c2r, c2i = cmul('r2r','r2i','w2r','w2i')
    c3r, c3i = cmul('r3r','r3i','w3r','w3i')

    return f"""    for (size_t k = 0; k < K; k += {VL}) {{
        const {V} x0r = R4_LD(&in_re[0*K+k]), x0i = R4_LD(&in_im[0*K+k]);
        const {V} r1r = R4_LD(&in_re[1*K+k]), r1i = R4_LD(&in_im[1*K+k]);
        const {V} r2r = R4_LD(&in_re[2*K+k]), r2i = R4_LD(&in_im[2*K+k]);
        const {V} r3r = R4_LD(&in_re[3*K+k]), r3i = R4_LD(&in_im[3*K+k]);
        const {V} w1r = R4_LD(&tw_re[0*K+k]), w1i = R4_LD(&tw_im[0*K+k]);
        const {V} w2r = R4_LD(&tw_re[1*K+k]), w2i = R4_LD(&tw_im[1*K+k]);
        const {V} w3r = R4_LD(&tw_re[2*K+k]), w3i = R4_LD(&tw_im[2*K+k]);
        const {V} x1r = {c1r}, x1i = {c1i};
        const {V} x2r = {c2r}, x2i = {c2i};
        const {V} x3r = {c3r}, x3i = {c3i};
        const {V} t0r = {ADD}(x0r, x2r), t0i = {ADD}(x0i, x2i);
        const {V} t1r = {SUB}(x0r, x2r), t1i = {SUB}(x0i, x2i);
        const {V} t2r = {ADD}(x1r, x3r), t2i = {ADD}(x1i, x3i);
        const {V} t3r = {SUB}(x1r, x3r), t3i = {SUB}(x1i, x3i);
        R4_ST(&out_re[0*K+k], {ADD}(t0r, t2r)); R4_ST(&out_im[0*K+k], {ADD}(t0i, t2i));
        R4_ST(&out_re[2*K+k], {SUB}(t0r, t2r)); R4_ST(&out_im[2*K+k], {SUB}(t0i, t2i));
        R4_ST(&out_re[1*K+k], {r1()}); R4_ST(&out_im[1*K+k], {i1()});
        R4_ST(&out_re[3*K+k], {r3()}); R4_ST(&out_im[3*K+k], {i3()});
    }}"""

def gen_scalar_dif_tw(direction):
    """DIF fused: butterfly → post-twiddle outputs 1..3 → store, one pass."""
    fwd = direction == 'fwd'
    if fwd:
        r1 = 't1r + t3i'; i1 = 't1i - t3r'
        r3 = 't1r - t3i'; i3 = 't1i + t3r'
        cm = lambda vr,vi,wr,wi: (f'{vr}*{wr} - {vi}*{wi}', f'{vr}*{wi} + {vi}*{wr}')
    else:
        r1 = 't1r - t3i'; i1 = 't1i + t3r'
        r3 = 't1r + t3i'; i3 = 't1i - t3r'
        cm = lambda vr,vi,wr,wi: (f'{vr}*{wr} + {vi}*{wi}', f'-{vr}*{wi} + {vi}*{wr}')
    y1r, y1i = cm(r1, i1, 'w1r', 'w1i')
    y2r, y2i = cm('(t0r - t2r)', '(t0i - t2i)', 'w2r', 'w2i')
    y3r, y3i = cm(r3, i3, 'w3r', 'w3i')
    return f"""    for (size_t k = 0; k < K; k++) {{
        const double x0r = in_re[0*K+k], x0i = in_im[0*K+k];
        const double x1r = in_re[1*K+k], x1i = in_im[1*K+k];
        const double x2r = in_re[2*K+k], x2i = in_im[2*K+k];
        const double x3r = in_re[3*K+k], x3i = in_im[3*K+k];
        const double t0r = x0r + x2r, t0i = x0i + x2i;
        const double t1r = x0r - x2r, t1i = x0i - x2i;
        const double t2r = x1r + x3r, t2i = x1i + x3i;
        const double t3r = x1r - x3r, t3i = x1i - x3i;
        const double w1r = tw_re[0*K+k], w1i = tw_im[0*K+k];
        const double w2r = tw_re[1*K+k], w2i = tw_im[1*K+k];
        const double w3r = tw_re[2*K+k], w3i = tw_im[2*K+k];
        out_re[0*K+k] = t0r + t2r; out_im[0*K+k] = t0i + t2i;
        out_re[1*K+k] = {y1r}; out_im[1*K+k] = {y1i};
        out_re[2*K+k] = {y2r}; out_im[2*K+k] = {y2i};
        out_re[3*K+k] = {y3r}; out_im[3*K+k] = {y3i};
    }}"""

def gen_simd_dif_tw(isa_name, direction):
    """DIF fused: butterfly → post-twiddle → store, one pass."""
    I = ISA[isa_name]
    V, p, VL = I['V'], I['p'], I['VL']
    ADD, SUB = f'{p}_add_pd', f'{p}_sub_pd'
    MUL = f'{p}_mul_pd'
    FNMA = f'{p}_fnmadd_pd'
    FMA  = f'{p}_fmadd_pd'
    fwd = direction == 'fwd'
    if fwd:
        def cmul(xr,xi,wr,wi): return (f'{FNMA}({xi},{wi},{MUL}({xr},{wr}))',
                                        f'{FMA}({xr},{wi},{MUL}({xi},{wr}))')
        r1, i1 = f'{ADD}(t1r, t3i)', f'{SUB}(t1i, t3r)'
        r3, i3 = f'{SUB}(t1r, t3i)', f'{ADD}(t1i, t3r)'
    else:
        def cmul(xr,xi,wr,wi): return (f'{FMA}({xi},{wi},{MUL}({xr},{wr}))',
                                        f'{FNMA}({xr},{wi},{MUL}({xi},{wr}))')
        r1, i1 = f'{SUB}(t1r, t3i)', f'{ADD}(t1i, t3r)'
        r3, i3 = f'{ADD}(t1r, t3i)', f'{SUB}(t1i, t3r)'
    y1r, y1i = cmul(r1, i1, 'w1r', 'w1i')
    y2r, y2i = cmul(f'{SUB}(t0r, t2r)', f'{SUB}(t0i, t2i)', 'w2r', 'w2i')
    y3r, y3i = cmul(r3, i3, 'w3r', 'w3i')

    return f"""    for (size_t k = 0; k < K; k += {VL}) {{
        const {V} x0r = R4_LD(&in_re[0*K+k]), x0i = R4_LD(&in_im[0*K+k]);
        const {V} x1r = R4_LD(&in_re[1*K+k]), x1i = R4_LD(&in_im[1*K+k]);
        const {V} x2r = R4_LD(&in_re[2*K+k]), x2i = R4_LD(&in_im[2*K+k]);
        const {V} x3r = R4_LD(&in_re[3*K+k]), x3i = R4_LD(&in_im[3*K+k]);
        const {V} t0r = {ADD}(x0r, x2r), t0i = {ADD}(x0i, x2i);
        const {V} t1r = {SUB}(x0r, x2r), t1i = {SUB}(x0i, x2i);
        const {V} t2r = {ADD}(x1r, x3r), t2i = {ADD}(x1i, x3i);
        const {V} t3r = {SUB}(x1r, x3r), t3i = {SUB}(x1i, x3i);
        const {V} w1r = R4_LD(&tw_re[0*K+k]), w1i = R4_LD(&tw_im[0*K+k]);
        const {V} w2r = R4_LD(&tw_re[1*K+k]), w2i = R4_LD(&tw_im[1*K+k]);
        const {V} w3r = R4_LD(&tw_re[2*K+k]), w3i = R4_LD(&tw_im[2*K+k]);
        R4_ST(&out_re[0*K+k], {ADD}(t0r, t2r)); R4_ST(&out_im[0*K+k], {ADD}(t0i, t2i));
        R4_ST(&out_re[1*K+k], {y1r}); R4_ST(&out_im[1*K+k], {y1i});
        R4_ST(&out_re[2*K+k], {y2r}); R4_ST(&out_im[2*K+k], {y2i});
        R4_ST(&out_re[3*K+k], {y3r}); R4_ST(&out_im[3*K+k], {y3i});
    }}"""


# ═══════════════════════════════════════════════════════════════
# U=2: two k-groups per iteration for ILP + cache miss hiding
#
# Register budget (AVX-512):
#   notw U=2:  2×8 data + 8 temp = 24 ZMM (8 free)
#   tw U=2:    2×8 data + 2×6 tw + 8 temp = 36 → over 32!
#              → share twiddle loads? No, different k offsets.
#              → process A butterfly, store, then B. Compiler interleaves.
#              Peak: 8 data + 6 tw + 8 temp + 8 data(B prefetch) = 30 ZMM.
#   dif_tw U=2: same budget as tw U=2.
#
# AVX2 (16 YMM):
#   notw U=2:  2×8 + 8 = 24 → over 16!  Not feasible.
#   Only AVX-512 gets U=2.
# ═══════════════════════════════════════════════════════════════

def _u2_body(gen_u1_fn, isa_name, direction):
    """Take a U=1 generator, return U=2 body with two scoped blocks per iteration.
    
    The compiler sees two independent blocks and interleaves their operations
    across both FMA ports. This is equivalent to manual interleaving but
    simpler and equally effective at -O3.
    """
    I = ISA[isa_name]
    VL = I['VL']
    # Get the U=1 body and extract the loop interior
    u1 = gen_u1_fn(isa_name, direction)
    # Extract the lines between the for-loop braces
    lines = u1.strip().split('\n')
    # lines[0] = "    for (size_t k = 0; k < K; k += VL) {"
    # lines[-1] = "    }"
    # body = lines[1:-1]
    body_a = '\n'.join(lines[1:-1])
    # Create body B by replacing k with k+VL in load/store addresses
    body_b = body_a.replace('+k]', f'+k+{VL}]').replace('+k)', f'+k+{VL})')

    return f"""    for (size_t k = 0; k < K; k += {VL*2}) {{
        {{ /* Pipeline A: k */
{body_a}
        }}
        {{ /* Pipeline B: k+{VL} */
{body_b}
        }}
    }}"""

def gen_simd_notw_u2(isa_name, direction):
    return _u2_body(gen_simd_notw, isa_name, direction)

def gen_simd_tw_u2(isa_name, direction):
    return _u2_body(gen_simd_tw, isa_name, direction)

def gen_simd_dif_tw_u2(isa_name, direction):
    return _u2_body(gen_simd_dif_tw, isa_name, direction)

def gen_stats_header(isa_name):
    """Generate operation count table for header comment."""
    is_scalar = isa_name == 'scalar'
    # R=4 butterfly: 8 add + 8 sub = 16 arith, 0 mul
    # DIT tw adds: 3 cmuls × (1 fnma + 1 fma + 2 mul) = 3 fnma + 3 fma + 6 mul (SIMD)
    #              or 3 cmuls × (4 mul + 2 add/sub) for scalar
    # DIF tw same op count as DIT tw (fused: butterfly + 3 post-cmuls)
    # Loads: notw=8, tw=8+6=14, dif_tw=8+6=14
    # Stores: 8 always
    if is_scalar:
        lines = [
            ' *   kernel                 add   sub   mul   neg   fma   fms | arith flops |  ld  st  mem',
            ' *   ──────────────────── ───── ───── ───── ───── ───── ───── + ───── ───── + ─── ─── ────',
            ' *   notw fwd/bwd              8     8     0     0     0     0 |    16    16 |   8   8   16',
            ' *   dit_tw fwd/bwd            8     8    12     0     0     0 |    28    28 |  14   8   22',
            ' *   dif_tw fwd/bwd            8     8    12     0     0     0 |    28    28 |  14   8   22',
        ]
    else:
        lines = [
            ' *   kernel                 add   sub   mul   neg  fnma   fma | arith flops |  ld  st  mem',
            ' *   ──────────────────── ───── ───── ───── ───── ───── ───── + ───── ───── + ─── ─── ────',
            ' *   notw fwd/bwd              8     8     0     0     0     0 |    16    16 |   8   8   16',
            ' *   dit_tw fwd/bwd            8     8     6     0     3     3 |    28    34 |  14   8   22',
            ' *   dif_tw fwd/bwd            8     8     6     0     3     3 |    28    34 |  14   8   22',
        ]
    return '\n'.join(lines)

def gen_file(isa_name):
    I = ISA[isa_name]
    guard = f'FFT_RADIX4_{isa_name.upper()}_H'
    is_scalar = isa_name == 'scalar'

    parts = []
    parts.append(f'''/**
 * @file fft_radix4_{isa_name}.h
 * @brief DFT-4 {isa_name.upper()} codelets (notw + DIT tw + DIF tw)
 *
 * DFT-4 = 2xDFT-2 + x(-j) combine. 16 adds, 0 muls, 0 constants.
 * Forward/backward differ only in the xj sign on the odd-diff term.
 * Twiddle table: 3*K doubles per component (W^1, W^2, W^3).
 *
 * ── Operation counts per k-step ──
 *
{gen_stats_header(isa_name)}
 *
 * Generated by gen_radix4.py — do not edit.
 */

#ifndef {guard}
#define {guard}

{I['guard_begin']}

#include <stddef.h>''')

    if not is_scalar:
        parts.append('#include <immintrin.h>')
        parts.append('')
        parts.append(f'#define R4_LD(p) {I["p"]}_loadu_pd(p)')
        parts.append(f'#define R4_ST(p,v) {I["p"]}_storeu_pd((p),(v))')

    for direction in ['fwd', 'bwd']:
        # notw
        parts.append(f'''
{I['target']}
static inline void
radix4_notw_dit_kernel_{direction}_{isa_name}(
    const double * __restrict__ in_re, const double * __restrict__ in_im,
    double * __restrict__ out_re, double * __restrict__ out_im,
    size_t K)
{{''')
        if is_scalar:
            parts.append(gen_scalar_notw(direction))
        else:
            parts.append(gen_simd_notw(isa_name, direction))
        parts.append('}')

        # DIT tw
        parts.append(f'''
{I['target']}
static inline void
radix4_tw_dit_kernel_{direction}_{isa_name}(
    const double * __restrict__ in_re, const double * __restrict__ in_im,
    double * __restrict__ out_re, double * __restrict__ out_im,
    const double * __restrict__ tw_re, const double * __restrict__ tw_im,
    size_t K)
{{''')
        if is_scalar:
            parts.append(gen_scalar_tw(direction))
        else:
            parts.append(gen_simd_tw(isa_name, direction))
        parts.append('}')

        # DIF tw (fused single-pass)
        parts.append(f'''
{I['target']}
static inline void
radix4_tw_dif_kernel_{direction}_{isa_name}(
    const double * __restrict__ in_re, const double * __restrict__ in_im,
    double * __restrict__ out_re, double * __restrict__ out_im,
    const double * __restrict__ tw_re, const double * __restrict__ tw_im,
    size_t K)
{{''')
        if is_scalar:
            parts.append(gen_scalar_dif_tw(direction))
        else:
            parts.append(gen_simd_dif_tw(isa_name, direction))
        parts.append('}')

    # U=2 variants (AVX-512 only — AVX2 doesn't have register budget)
    if isa_name == 'avx512':
        parts.append('\n/* === U=2: two k-groups per iteration (ILP + cache miss hiding) === */')
        for direction in ['fwd', 'bwd']:
            parts.append(f'''
{I['target']}
static inline void
radix4_notw_dit_kernel_{direction}_{isa_name}_u2(
    const double * __restrict__ in_re, const double * __restrict__ in_im,
    double * __restrict__ out_re, double * __restrict__ out_im,
    size_t K)
{{''')
            parts.append(gen_simd_notw_u2(isa_name, direction))
            parts.append('}')

            parts.append(f'''
{I['target']}
static inline void
radix4_tw_dit_kernel_{direction}_{isa_name}_u2(
    const double * __restrict__ in_re, const double * __restrict__ in_im,
    double * __restrict__ out_re, double * __restrict__ out_im,
    const double * __restrict__ tw_re, const double * __restrict__ tw_im,
    size_t K)
{{''')
            parts.append(gen_simd_tw_u2(isa_name, direction))
            parts.append('}')

            parts.append(f'''
{I['target']}
static inline void
radix4_tw_dif_kernel_{direction}_{isa_name}_u2(
    const double * __restrict__ in_re, const double * __restrict__ in_im,
    double * __restrict__ out_re, double * __restrict__ out_im,
    const double * __restrict__ tw_re, const double * __restrict__ tw_im,
    size_t K)
{{''')
            parts.append(gen_simd_dif_tw_u2(isa_name, direction))
            parts.append('}')

    # ── sv codelets (SIMD only): no loop, elements at stride vs ──
    if not is_scalar:
        parts.append('\n/* === sv codelets: no loop, elements at stride vs === */')
        parts.append(f'/* Executor calls K/{I["VL"]} times, advancing base pointers by {I["VL"]}. */')

        V, p, VL = I['V'], I['p'], I['VL']
        ADD, SUB = f'{p}_add_pd', f'{p}_sub_pd'
        MUL = f'{p}_mul_pd'
        FNMA = f'{p}_fnmadd_pd'
        FMA = f'{p}_fmadd_pd'

        for direction in ['fwd', 'bwd']:
            fwd = direction == 'fwd'
            if fwd:
                j_r1, j_i1 = f'{ADD}(t1r, t3i)', f'{SUB}(t1i, t3r)'
                j_r3, j_i3 = f'{SUB}(t1r, t3i)', f'{ADD}(t1i, t3r)'
            else:
                j_r1, j_i1 = f'{SUB}(t1r, t3i)', f'{ADD}(t1i, t3r)'
                j_r3, j_i3 = f'{ADD}(t1r, t3i)', f'{SUB}(t1i, t3r)'

            if fwd:
                def cmul_sv(xr,xi,wr,wi): return (f'{FNMA}({xi},{wi},{MUL}({xr},{wr}))',
                                                    f'{FMA}({xr},{wi},{MUL}({xi},{wr}))')
            else:
                def cmul_sv(xr,xi,wr,wi): return (f'{FMA}({xi},{wi},{MUL}({xr},{wr}))',
                                                    f'{FNMA}({xr},{wi},{MUL}({xi},{wr}))')

            # n1sv: no twiddle
            parts.append(f'''
{I['target']}
static inline void
radix4_n1sv_kernel_{direction}_{isa_name}(
    const double * __restrict__ in_re, const double * __restrict__ in_im,
    double * __restrict__ out_re, double * __restrict__ out_im,
    size_t vs)
{{
    const {V} x0r = R4_LD(&in_re[0*vs]), x0i = R4_LD(&in_im[0*vs]);
    const {V} x1r = R4_LD(&in_re[1*vs]), x1i = R4_LD(&in_im[1*vs]);
    const {V} x2r = R4_LD(&in_re[2*vs]), x2i = R4_LD(&in_im[2*vs]);
    const {V} x3r = R4_LD(&in_re[3*vs]), x3i = R4_LD(&in_im[3*vs]);
    const {V} t0r = {ADD}(x0r, x2r), t0i = {ADD}(x0i, x2i);
    const {V} t1r = {SUB}(x0r, x2r), t1i = {SUB}(x0i, x2i);
    const {V} t2r = {ADD}(x1r, x3r), t2i = {ADD}(x1i, x3i);
    const {V} t3r = {SUB}(x1r, x3r), t3i = {SUB}(x1i, x3i);
    R4_ST(&out_re[0*vs], {ADD}(t0r, t2r)); R4_ST(&out_im[0*vs], {ADD}(t0i, t2i));
    R4_ST(&out_re[2*vs], {SUB}(t0r, t2r)); R4_ST(&out_im[2*vs], {SUB}(t0i, t2i));
    R4_ST(&out_re[1*vs], {j_r1}); R4_ST(&out_im[1*vs], {j_i1});
    R4_ST(&out_re[3*vs], {j_r3}); R4_ST(&out_im[3*vs], {j_i3});
}}''')

            # t1sv DIT: twiddle input, then butterfly
            c1r, c1i = cmul_sv('r1r','r1i','w1r','w1i')
            c2r, c2i = cmul_sv('r2r','r2i','w2r','w2i')
            c3r, c3i = cmul_sv('r3r','r3i','w3r','w3i')
            parts.append(f'''
{I['target']}
static inline void
radix4_t1sv_dit_kernel_{direction}_{isa_name}(
    const double * __restrict__ in_re, const double * __restrict__ in_im,
    double * __restrict__ out_re, double * __restrict__ out_im,
    const double * __restrict__ tw_re, const double * __restrict__ tw_im,
    size_t vs)
{{
    const {V} x0r = R4_LD(&in_re[0*vs]), x0i = R4_LD(&in_im[0*vs]);
    const {V} r1r = R4_LD(&in_re[1*vs]), r1i = R4_LD(&in_im[1*vs]);
    const {V} r2r = R4_LD(&in_re[2*vs]), r2i = R4_LD(&in_im[2*vs]);
    const {V} r3r = R4_LD(&in_re[3*vs]), r3i = R4_LD(&in_im[3*vs]);
    const {V} w1r = R4_LD(&tw_re[0*vs]), w1i = R4_LD(&tw_im[0*vs]);
    const {V} w2r = R4_LD(&tw_re[1*vs]), w2i = R4_LD(&tw_im[1*vs]);
    const {V} w3r = R4_LD(&tw_re[2*vs]), w3i = R4_LD(&tw_im[2*vs]);
    const {V} x1r = {c1r}, x1i = {c1i};
    const {V} x2r = {c2r}, x2i = {c2i};
    const {V} x3r = {c3r}, x3i = {c3i};
    const {V} t0r = {ADD}(x0r, x2r), t0i = {ADD}(x0i, x2i);
    const {V} t1r = {SUB}(x0r, x2r), t1i = {SUB}(x0i, x2i);
    const {V} t2r = {ADD}(x1r, x3r), t2i = {ADD}(x1i, x3i);
    const {V} t3r = {SUB}(x1r, x3r), t3i = {SUB}(x1i, x3i);
    R4_ST(&out_re[0*vs], {ADD}(t0r, t2r)); R4_ST(&out_im[0*vs], {ADD}(t0i, t2i));
    R4_ST(&out_re[2*vs], {SUB}(t0r, t2r)); R4_ST(&out_im[2*vs], {SUB}(t0i, t2i));
    R4_ST(&out_re[1*vs], {j_r1}); R4_ST(&out_im[1*vs], {j_i1});
    R4_ST(&out_re[3*vs], {j_r3}); R4_ST(&out_im[3*vs], {j_i3});
}}''')

            # t1sv DIF: butterfly, then twiddle output
            y1r_sv, y1i_sv = cmul_sv(j_r1, j_i1, 'w1r', 'w1i')
            y2r_sv, y2i_sv = cmul_sv(f'{SUB}(t0r, t2r)', f'{SUB}(t0i, t2i)', 'w2r', 'w2i')
            y3r_sv, y3i_sv = cmul_sv(j_r3, j_i3, 'w3r', 'w3i')
            parts.append(f'''
{I['target']}
static inline void
radix4_t1sv_dif_kernel_{direction}_{isa_name}(
    const double * __restrict__ in_re, const double * __restrict__ in_im,
    double * __restrict__ out_re, double * __restrict__ out_im,
    const double * __restrict__ tw_re, const double * __restrict__ tw_im,
    size_t vs)
{{
    const {V} x0r = R4_LD(&in_re[0*vs]), x0i = R4_LD(&in_im[0*vs]);
    const {V} x1r = R4_LD(&in_re[1*vs]), x1i = R4_LD(&in_im[1*vs]);
    const {V} x2r = R4_LD(&in_re[2*vs]), x2i = R4_LD(&in_im[2*vs]);
    const {V} x3r = R4_LD(&in_re[3*vs]), x3i = R4_LD(&in_im[3*vs]);
    const {V} t0r = {ADD}(x0r, x2r), t0i = {ADD}(x0i, x2i);
    const {V} t1r = {SUB}(x0r, x2r), t1i = {SUB}(x0i, x2i);
    const {V} t2r = {ADD}(x1r, x3r), t2i = {ADD}(x1i, x3i);
    const {V} t3r = {SUB}(x1r, x3r), t3i = {SUB}(x1i, x3i);
    const {V} w1r = R4_LD(&tw_re[0*vs]), w1i = R4_LD(&tw_im[0*vs]);
    const {V} w2r = R4_LD(&tw_re[1*vs]), w2i = R4_LD(&tw_im[1*vs]);
    const {V} w3r = R4_LD(&tw_re[2*vs]), w3i = R4_LD(&tw_im[2*vs]);
    R4_ST(&out_re[0*vs], {ADD}(t0r, t2r)); R4_ST(&out_im[0*vs], {ADD}(t0i, t2i));
    R4_ST(&out_re[1*vs], {y1r_sv}); R4_ST(&out_im[1*vs], {y1i_sv});
    R4_ST(&out_re[2*vs], {y2r_sv}); R4_ST(&out_im[2*vs], {y2i_sv});
    R4_ST(&out_re[3*vs], {y3r_sv}); R4_ST(&out_im[3*vs], {y3i_sv});
}}''')

    # ── FFTW-style codelets: n1 (separate is/os) + t1 (in-place twiddle) ──
    parts.append('\n/* ================================================================')
    parts.append(' * FFTW-style codelets for recursive CT executor')
    parts.append(' *')
    parts.append(' * n1:  out-of-place, separate input/output strides (is, os).')
    parts.append(' *      Vectorized over vl batch dimension (ivs=ovs=1 for SIMD).')
    parts.append(' *      This is the child DFT in CT: reads decimated input, writes contiguous output.')
    parts.append(' *')
    parts.append(' * t1:  in-place twiddle + butterfly. Single stride ios between butterfly legs.')
    parts.append(' *      Loops m from mb to me with stride ms.')
    parts.append(' *      This runs after n1 on the contiguous output.')
    parts.append(' * ================================================================ */\n')

    for direction in ['fwd', 'bwd']:
        fwd = direction == 'fwd'
        if fwd:
            bfly_r1 = 't1r + t3i'; bfly_i1 = 't1i - t3r'
            bfly_r3 = 't1r - t3i'; bfly_i3 = 't1i + t3r'
        else:
            bfly_r1 = 't1r - t3i'; bfly_i1 = 't1i + t3r'
            bfly_r3 = 't1r + t3i'; bfly_i3 = 't1i - t3r'

        # ── scalar n1 ──
        parts.append(f'''
static inline void
radix4_n1_{direction}(
    const double * __restrict__ in_re, const double * __restrict__ in_im,
    double * __restrict__ out_re, double * __restrict__ out_im,
    size_t is, size_t os, size_t vl, size_t ivs, size_t ovs)
{{
    for (size_t k = 0; k < vl; k++) {{
        const double x0r = in_re[0*is + k*ivs], x0i = in_im[0*is + k*ivs];
        const double x1r = in_re[1*is + k*ivs], x1i = in_im[1*is + k*ivs];
        const double x2r = in_re[2*is + k*ivs], x2i = in_im[2*is + k*ivs];
        const double x3r = in_re[3*is + k*ivs], x3i = in_im[3*is + k*ivs];
        const double t0r = x0r + x2r, t0i = x0i + x2i;
        const double t1r = x0r - x2r, t1i = x0i - x2i;
        const double t2r = x1r + x3r, t2i = x1i + x3i;
        const double t3r = x1r - x3r, t3i = x1i - x3i;
        out_re[0*os + k*ovs] = t0r + t2r; out_im[0*os + k*ovs] = t0i + t2i;
        out_re[2*os + k*ovs] = t0r - t2r; out_im[2*os + k*ovs] = t0i - t2i;
        out_re[1*os + k*ovs] = {bfly_r1}; out_im[1*os + k*ovs] = {bfly_i1};
        out_re[3*os + k*ovs] = {bfly_r3}; out_im[3*os + k*ovs] = {bfly_i3};
    }}
}}''')

        # ── scalar t1 DIT (in-place: twiddle inputs 1..R-1, then butterfly) ──
        if fwd:
            cm_s = lambda vr,vi,wr,wi: (f'{vr}*{wr} - {vi}*{wi}', f'{vr}*{wi} + {vi}*{wr}')
        else:
            cm_s = lambda vr,vi,wr,wi: (f'{vr}*{wr} + {vi}*{wi}', f'-{vr}*{wi} + {vi}*{wr}')
        sc1r, sc1i = cm_s('r1r','r1i','w1r','w1i')
        sc2r, sc2i = cm_s('r2r','r2i','w2r','w2i')
        sc3r, sc3i = cm_s('r3r','r3i','w3r','w3i')
        parts.append(f'''
static inline void
radix4_t1_dit_{direction}(
    double * __restrict__ rio_re, double * __restrict__ rio_im,
    const double * __restrict__ W_re, const double * __restrict__ W_im,
    size_t ios, size_t mb, size_t me, size_t ms)
{{
    for (size_t m = mb; m < me; m++) {{
        const double x0r = rio_re[m*ms + 0*ios], x0i = rio_im[m*ms + 0*ios];
        const double r1r = rio_re[m*ms + 1*ios], r1i = rio_im[m*ms + 1*ios];
        const double r2r = rio_re[m*ms + 2*ios], r2i = rio_im[m*ms + 2*ios];
        const double r3r = rio_re[m*ms + 3*ios], r3i = rio_im[m*ms + 3*ios];
        const double w1r = W_re[0*me + m], w1i = W_im[0*me + m];
        const double w2r = W_re[1*me + m], w2i = W_im[1*me + m];
        const double w3r = W_re[2*me + m], w3i = W_im[2*me + m];
        const double x1r = {sc1r}, x1i = {sc1i};
        const double x2r = {sc2r}, x2i = {sc2i};
        const double x3r = {sc3r}, x3i = {sc3i};
        const double t0r = x0r + x2r, t0i = x0i + x2i;
        const double t1r = x0r - x2r, t1i = x0i - x2i;
        const double t2r = x1r + x3r, t2i = x1i + x3i;
        const double t3r = x1r - x3r, t3i = x1i - x3i;
        rio_re[m*ms + 0*ios] = t0r + t2r; rio_im[m*ms + 0*ios] = t0i + t2i;
        rio_re[m*ms + 2*ios] = t0r - t2r; rio_im[m*ms + 2*ios] = t0i - t2i;
        rio_re[m*ms + 1*ios] = {bfly_r1}; rio_im[m*ms + 1*ios] = {bfly_i1};
        rio_re[m*ms + 3*ios] = {bfly_r3}; rio_im[m*ms + 3*ios] = {bfly_i3};
    }}
}}''')

        # ── scalar t1 DIT log3 (derive w2, w3 from w1) ──
        # For scalar: w2 = w1*w1, w3 = w2*w1 (standard cmul)
        if fwd:
            cm_log3 = lambda ar,ai,br,bi: (f'{ar}*{br} - {ai}*{bi}', f'{ar}*{bi} + {ai}*{br}')
        else:
            cm_log3 = lambda ar,ai,br,bi: (f'{ar}*{br} - {ai}*{bi}', f'{ar}*{bi} + {ai}*{br}')
        parts.append(f'''
static inline void
radix4_t1_dit_log3_{direction}(
    double * __restrict__ rio_re, double * __restrict__ rio_im,
    const double * __restrict__ W_re, const double * __restrict__ W_im,
    size_t ios, size_t mb, size_t me, size_t ms)
{{
    for (size_t m = mb; m < me; m++) {{
        const double x0r = rio_re[m*ms + 0*ios], x0i = rio_im[m*ms + 0*ios];
        const double r1r = rio_re[m*ms + 1*ios], r1i = rio_im[m*ms + 1*ios];
        const double r2r = rio_re[m*ms + 2*ios], r2i = rio_im[m*ms + 2*ios];
        const double r3r = rio_re[m*ms + 3*ios], r3i = rio_im[m*ms + 3*ios];
        const double w1r = W_re[0*me + m], w1i = W_im[0*me + m];
        const double w2r = w1r*w1r - w1i*w1i, w2i = 2.0*w1r*w1i;
        const double w3r = w2r*w1r - w2i*w1i, w3i = w2r*w1i + w2i*w1r;
        const double x1r = {sc1r}, x1i = {sc1i};
        const double x2r = {sc2r}, x2i = {sc2i};
        const double x3r = {sc3r}, x3i = {sc3i};
        const double t0r = x0r + x2r, t0i = x0i + x2i;
        const double t1r = x0r - x2r, t1i = x0i - x2i;
        const double t2r = x1r + x3r, t2i = x1i + x3i;
        const double t3r = x1r - x3r, t3i = x1i - x3i;
        rio_re[m*ms + 0*ios] = t0r + t2r; rio_im[m*ms + 0*ios] = t0i + t2i;
        rio_re[m*ms + 2*ios] = t0r - t2r; rio_im[m*ms + 2*ios] = t0i - t2i;
        rio_re[m*ms + 1*ios] = {bfly_r1}; rio_im[m*ms + 1*ios] = {bfly_i1};
        rio_re[m*ms + 3*ios] = {bfly_r3}; rio_im[m*ms + 3*ios] = {bfly_i3};
    }}
}}''')

        # ── scalar t1 DIF (in-place: butterfly, then twiddle outputs 1..R-1) ──
        if fwd:
            dif_cm = lambda vr,vi,wr,wi: (f'{vr}*{wr} - {vi}*{wi}', f'{vr}*{wi} + {vi}*{wr}')
        else:
            dif_cm = lambda vr,vi,wr,wi: (f'{vr}*{wr} + {vi}*{wi}', f'-{vr}*{wi} + {vi}*{wr}')
        dy1r, dy1i = dif_cm(bfly_r1, bfly_i1, 'w1r', 'w1i')
        dy2r, dy2i = dif_cm('(t0r - t2r)', '(t0i - t2i)', 'w2r', 'w2i')
        dy3r, dy3i = dif_cm(bfly_r3, bfly_i3, 'w3r', 'w3i')
        parts.append(f'''
static inline void
radix4_t1_dif_{direction}(
    double * __restrict__ rio_re, double * __restrict__ rio_im,
    const double * __restrict__ W_re, const double * __restrict__ W_im,
    size_t ios, size_t mb, size_t me, size_t ms)
{{
    for (size_t m = mb; m < me; m++) {{
        const double x0r = rio_re[m*ms + 0*ios], x0i = rio_im[m*ms + 0*ios];
        const double x1r = rio_re[m*ms + 1*ios], x1i = rio_im[m*ms + 1*ios];
        const double x2r = rio_re[m*ms + 2*ios], x2i = rio_im[m*ms + 2*ios];
        const double x3r = rio_re[m*ms + 3*ios], x3i = rio_im[m*ms + 3*ios];
        const double t0r = x0r + x2r, t0i = x0i + x2i;
        const double t1r = x0r - x2r, t1i = x0i - x2i;
        const double t2r = x1r + x3r, t2i = x1i + x3i;
        const double t3r = x1r - x3r, t3i = x1i - x3i;
        const double w1r = W_re[0*me + m], w1i = W_im[0*me + m];
        const double w2r = W_re[1*me + m], w2i = W_im[1*me + m];
        const double w3r = W_re[2*me + m], w3i = W_im[2*me + m];
        rio_re[m*ms + 0*ios] = t0r + t2r; rio_im[m*ms + 0*ios] = t0i + t2i;
        rio_re[m*ms + 1*ios] = {dy1r}; rio_im[m*ms + 1*ios] = {dy1i};
        rio_re[m*ms + 2*ios] = {dy2r}; rio_im[m*ms + 2*ios] = {dy2i};
        rio_re[m*ms + 3*ios] = {dy3r}; rio_im[m*ms + 3*ios] = {dy3i};
    }}
}}''')

        # ── scalar t1_oop DIT (out-of-place: twiddle inputs 1..R-1, then butterfly) ──
        parts.append(f'''
static inline void
radix4_t1_oop_dit_{direction}_scalar(
    const double * __restrict__ in_re, const double * __restrict__ in_im,
    double * __restrict__ out_re, double * __restrict__ out_im,
    const double * __restrict__ W_re, const double * __restrict__ W_im,
    size_t is, size_t os, size_t me)
{{
    for (size_t m = 0; m < me; m++) {{
        const double x0r = in_re[m + 0*is], x0i = in_im[m + 0*is];
        const double r1r = in_re[m + 1*is], r1i = in_im[m + 1*is];
        const double r2r = in_re[m + 2*is], r2i = in_im[m + 2*is];
        const double r3r = in_re[m + 3*is], r3i = in_im[m + 3*is];
        const double w1r = W_re[0*me + m], w1i = W_im[0*me + m];
        const double w2r = W_re[1*me + m], w2i = W_im[1*me + m];
        const double w3r = W_re[2*me + m], w3i = W_im[2*me + m];
        const double x1r = {sc1r}, x1i = {sc1i};
        const double x2r = {sc2r}, x2i = {sc2i};
        const double x3r = {sc3r}, x3i = {sc3i};
        const double t0r = x0r + x2r, t0i = x0i + x2i;
        const double t1r = x0r - x2r, t1i = x0i - x2i;
        const double t2r = x1r + x3r, t2i = x1i + x3i;
        const double t3r = x1r - x3r, t3i = x1i - x3i;
        out_re[m + 0*os] = t0r + t2r; out_im[m + 0*os] = t0i + t2i;
        out_re[m + 2*os] = t0r - t2r; out_im[m + 2*os] = t0i - t2i;
        out_re[m + 1*os] = {bfly_r1}; out_im[m + 1*os] = {bfly_i1};
        out_re[m + 3*os] = {bfly_r3}; out_im[m + 3*os] = {bfly_i3};
    }}
}}''')

        # ── scalar n1_scaled: n1 with output *= scale ──
        parts.append(f'''static inline void
radix4_n1_scaled_{direction}_scalar(
    const double * __restrict__ in_re, const double * __restrict__ in_im,
    double * __restrict__ out_re, double * __restrict__ out_im,
    size_t is, size_t os, size_t vl, double scale)
{{
    for (size_t m = 0; m < vl; m++) {{
        const double x0r = in_re[0*is+m], x0i = in_im[0*is+m];
        const double x1r = in_re[1*is+m], x1i = in_im[1*is+m];
        const double x2r = in_re[2*is+m], x2i = in_im[2*is+m];
        const double x3r = in_re[3*is+m], x3i = in_im[3*is+m];
        const double t0r = x0r + x2r, t0i = x0i + x2i;
        const double t1r = x0r - x2r, t1i = x0i - x2i;
        const double t2r = x1r + x3r, t2i = x1i + x3i;
        const double t3r = x1r - x3r, t3i = x1i - x3i;
        out_re[0*os+m] = scale * (t0r + t2r); out_im[0*os+m] = scale * (t0i + t2i);
        out_re[2*os+m] = scale * (t0r - t2r); out_im[2*os+m] = scale * (t0i - t2i);
        out_re[1*os+m] = scale * ({bfly_r1}); out_im[1*os+m] = scale * ({bfly_i1});
        out_re[3*os+m] = scale * ({bfly_r3}); out_im[3*os+m] = scale * ({bfly_i3});
    }}
}}''')
        parts.append(f'')

    # ── SIMD n1 + t1 (ivs=ovs=1 for n1, ms=1 for t1) ──
    if not is_scalar:
        V, p, VL = I['V'], I['p'], I['VL']
        ADD, SUB = f'{p}_add_pd', f'{p}_sub_pd'
        MUL = f'{p}_mul_pd'
        FNMA = f'{p}_fnmadd_pd'
        FMA  = f'{p}_fmadd_pd'

        for direction in ['fwd', 'bwd']:
            fwd = direction == 'fwd'
            if fwd:
                r1_e = f'{ADD}(t1r, t3i)'; i1_e = f'{SUB}(t1i, t3r)'
                r3_e = f'{SUB}(t1r, t3i)'; i3_e = f'{ADD}(t1i, t3r)'
                def cmul_v(xr,xi,wr,wi): return (f'{FNMA}({xi},{wi},{MUL}({xr},{wr}))',
                                                   f'{FMA}({xr},{wi},{MUL}({xi},{wr}))')
            else:
                r1_e = f'{SUB}(t1r, t3i)'; i1_e = f'{ADD}(t1i, t3r)'
                r3_e = f'{ADD}(t1r, t3i)'; i3_e = f'{SUB}(t1i, t3r)'
                def cmul_v(xr,xi,wr,wi): return (f'{FMA}({xi},{wi},{MUL}({xr},{wr}))',
                                                   f'{FNMA}({xr},{wi},{MUL}({xi},{wr}))')

            # ── SIMD n1: vectorize over vl (ivs=ovs=1) ──
            parts.append(f'''
{I['target']}
static inline void
radix4_n1_{direction}_{isa_name}(
    const double * __restrict__ in_re, const double * __restrict__ in_im,
    double * __restrict__ out_re, double * __restrict__ out_im,
    size_t is, size_t os, size_t vl)
{{
    /* SIMD n1: ivs=ovs=1 implied. Reads in_re[n*is+k], writes out_re[n*os+k]. */
    for (size_t k = 0; k < vl; k += {VL}) {{
        const {V} x0r = R4_LD(&in_re[0*is+k]), x0i = R4_LD(&in_im[0*is+k]);
        const {V} x1r = R4_LD(&in_re[1*is+k]), x1i = R4_LD(&in_im[1*is+k]);
        const {V} x2r = R4_LD(&in_re[2*is+k]), x2i = R4_LD(&in_im[2*is+k]);
        const {V} x3r = R4_LD(&in_re[3*is+k]), x3i = R4_LD(&in_im[3*is+k]);
        const {V} t0r = {ADD}(x0r, x2r), t0i = {ADD}(x0i, x2i);
        const {V} t1r = {SUB}(x0r, x2r), t1i = {SUB}(x0i, x2i);
        const {V} t2r = {ADD}(x1r, x3r), t2i = {ADD}(x1i, x3i);
        const {V} t3r = {SUB}(x1r, x3r), t3i = {SUB}(x1i, x3i);
        R4_ST(&out_re[0*os+k], {ADD}(t0r, t2r)); R4_ST(&out_im[0*os+k], {ADD}(t0i, t2i));
        R4_ST(&out_re[2*os+k], {SUB}(t0r, t2r)); R4_ST(&out_im[2*os+k], {SUB}(t0i, t2i));
        R4_ST(&out_re[1*os+k], {r1_e}); R4_ST(&out_im[1*os+k], {i1_e});
        R4_ST(&out_re[3*os+k], {r3_e}); R4_ST(&out_im[3*os+k], {i3_e});
    }}
}}''')

            # ── SIMD n1 with ovs: SIMD 4x4 transpose stores (AVX2 only) ──
            # Uses hard-coded _mm256_* unpacks for 4-lane vectors. AVX-512's
            # 8-lane vectors would need a distinct 8x4 transpose structure
            # (different intrinsics, different shuffle pattern). Rather than
            # writing a second implementation we gate to AVX2 — callers that
            # need a strided-store DIT on AVX-512 should use t1_oop_dit.
            if isa_name != 'avx2':
              pass  # skip emission; no-ops through to next block
            else:
              UP = '_mm256_unpacklo_pd'
              UH = '_mm256_unpackhi_pd'
              PM = '_mm256_permute2f128_pd'
              parts.append(f'''
{I['target']}
static inline void
radix4_n1_ovs_{direction}_{isa_name}(
    const double * __restrict__ in_re, const double * __restrict__ in_im,
    double * __restrict__ out_re, double * __restrict__ out_im,
    size_t is, size_t os, size_t vl, size_t ovs)
{{
    for (size_t k = 0; k < vl; k += {VL}) {{
        const {V} x0r = R4_LD(&in_re[0*is+k]), x0i = R4_LD(&in_im[0*is+k]);
        const {V} x1r = R4_LD(&in_re[1*is+k]), x1i = R4_LD(&in_im[1*is+k]);
        const {V} x2r = R4_LD(&in_re[2*is+k]), x2i = R4_LD(&in_im[2*is+k]);
        const {V} x3r = R4_LD(&in_re[3*is+k]), x3i = R4_LD(&in_im[3*is+k]);
        const {V} t0r = {ADD}(x0r, x2r), t0i = {ADD}(x0i, x2i);
        const {V} t1r = {SUB}(x0r, x2r), t1i = {SUB}(x0i, x2i);
        const {V} t2r = {ADD}(x1r, x3r), t2i = {ADD}(x1i, x3i);
        const {V} t3r = {SUB}(x1r, x3r), t3i = {SUB}(x1i, x3i);
        /* Butterfly results: y0=t0+t2, y2=t0-t2, y1, y3 */
        const {V} y0r = {ADD}(t0r, t2r), y0i = {ADD}(t0i, t2i);
        const {V} y2r = {SUB}(t0r, t2r), y2i = {SUB}(t0i, t2i);
        const {V} y1r = {r1_e}, y1i = {i1_e};
        const {V} y3r = {r3_e}, y3i = {i3_e};
        /* Transpose: [y0,y1,y2,y3] each has VL sub-seq values */
        {{ {V} lo01r={UP}(y0r,y1r), hi01r={UH}(y0r,y1r);
          {V} lo23r={UP}(y2r,y3r), hi23r={UH}(y2r,y3r);
          R4_ST(&out_re[(k+0)*ovs+os*0], {PM}(lo01r,lo23r,0x20));
          R4_ST(&out_re[(k+1)*ovs+os*0], {PM}(hi01r,hi23r,0x20));
          R4_ST(&out_re[(k+2)*ovs+os*0], {PM}(lo01r,lo23r,0x31));
          R4_ST(&out_re[(k+3)*ovs+os*0], {PM}(hi01r,hi23r,0x31));
        }}
        {{ {V} lo01i={UP}(y0i,y1i), hi01i={UH}(y0i,y1i);
          {V} lo23i={UP}(y2i,y3i), hi23i={UH}(y2i,y3i);
          R4_ST(&out_im[(k+0)*ovs+os*0], {PM}(lo01i,lo23i,0x20));
          R4_ST(&out_im[(k+1)*ovs+os*0], {PM}(hi01i,hi23i,0x20));
          R4_ST(&out_im[(k+2)*ovs+os*0], {PM}(lo01i,lo23i,0x31));
          R4_ST(&out_im[(k+3)*ovs+os*0], {PM}(hi01i,hi23i,0x31));
        }}
    }}
}}''')

            # ── SIMD t1 DIT: in-place, vectorize over m (ms=1 implied) ──
            vc1r, vc1i = cmul_v('r1r','r1i','w1r','w1i')
            vc2r, vc2i = cmul_v('r2r','r2i','w2r','w2i')
            vc3r, vc3i = cmul_v('r3r','r3i','w3r','w3i')
            parts.append(f'''
{I['target']}
static inline void
radix4_t1_dit_{direction}_{isa_name}(
    double * __restrict__ rio_re, double * __restrict__ rio_im,
    const double * __restrict__ W_re, const double * __restrict__ W_im,
    size_t ios, size_t me)
{{
    /* SIMD t1 DIT: ms=1, mb=0 implied. In-place twiddle + butterfly. */
    for (size_t m = 0; m < me; m += {VL}) {{
        {V} x0r = R4_LD(&rio_re[m + 0*ios]), x0i = R4_LD(&rio_im[m + 0*ios]);
        {V} r1r = R4_LD(&rio_re[m + 1*ios]), r1i = R4_LD(&rio_im[m + 1*ios]);
        {V} r2r = R4_LD(&rio_re[m + 2*ios]), r2i = R4_LD(&rio_im[m + 2*ios]);
        {V} r3r = R4_LD(&rio_re[m + 3*ios]), r3i = R4_LD(&rio_im[m + 3*ios]);
        const {V} w1r = R4_LD(&W_re[0*me+m]), w1i = R4_LD(&W_im[0*me+m]);
        const {V} w2r = R4_LD(&W_re[1*me+m]), w2i = R4_LD(&W_im[1*me+m]);
        const {V} w3r = R4_LD(&W_re[2*me+m]), w3i = R4_LD(&W_im[2*me+m]);
        const {V} x1r = {vc1r}, x1i = {vc1i};
        const {V} x2r = {vc2r}, x2i = {vc2i};
        const {V} x3r = {vc3r}, x3i = {vc3i};
        const {V} t0r = {ADD}(x0r, x2r), t0i = {ADD}(x0i, x2i);
        const {V} t1r = {SUB}(x0r, x2r), t1i = {SUB}(x0i, x2i);
        const {V} t2r = {ADD}(x1r, x3r), t2i = {ADD}(x1i, x3i);
        const {V} t3r = {SUB}(x1r, x3r), t3i = {SUB}(x1i, x3i);
        R4_ST(&rio_re[m + 0*ios], {ADD}(t0r, t2r)); R4_ST(&rio_im[m + 0*ios], {ADD}(t0i, t2i));
        R4_ST(&rio_re[m + 2*ios], {SUB}(t0r, t2r)); R4_ST(&rio_im[m + 2*ios], {SUB}(t0i, t2i));
        R4_ST(&rio_re[m + 1*ios], {r1_e}); R4_ST(&rio_im[m + 1*ios], {i1_e});
        R4_ST(&rio_re[m + 3*ios], {r3_e}); R4_ST(&rio_im[m + 3*ios], {i3_e});
    }}
}}''')

            # ── SIMD t1 DIT log3: load w1, derive w2=w1*w1, w3=w2*w1 ──
            # Saves 4 twiddle loads per iteration, costs 2 cmuls (4 FMA).
            # Wins when twiddle table overflows L1 (high K).
            parts.append(f'''
{I['target']}
static inline void
radix4_t1_dit_log3_{direction}_{isa_name}(
    double * __restrict__ rio_re, double * __restrict__ rio_im,
    const double * __restrict__ W_re, const double * __restrict__ W_im,
    size_t ios, size_t me)
{{
    /* SIMD t1 DIT log3: derive w2, w3 from w1. */
    for (size_t m = 0; m < me; m += {VL}) {{
        {V} x0r = R4_LD(&rio_re[m + 0*ios]), x0i = R4_LD(&rio_im[m + 0*ios]);
        {V} r1r = R4_LD(&rio_re[m + 1*ios]), r1i = R4_LD(&rio_im[m + 1*ios]);
        {V} r2r = R4_LD(&rio_re[m + 2*ios]), r2i = R4_LD(&rio_im[m + 2*ios]);
        {V} r3r = R4_LD(&rio_re[m + 3*ios]), r3i = R4_LD(&rio_im[m + 3*ios]);
        /* Load base twiddle w1, derive w2 = w1*w1, w3 = w2*w1 */
        const {V} w1r = R4_LD(&W_re[0*me+m]), w1i = R4_LD(&W_im[0*me+m]);
        const {V} w2r = {FNMA}(w1i, w1i, {MUL}(w1r, w1r));
        const {V} w2i = {FMA}(w1r, w1i, {MUL}(w1r, w1i));
        const {V} w3r = {FNMA}(w2i, w1i, {MUL}(w2r, w1r));
        const {V} w3i = {FMA}(w2r, w1i, {MUL}(w2i, w1r));
        const {V} x1r = {vc1r}, x1i = {vc1i};
        const {V} x2r = {vc2r}, x2i = {vc2i};
        const {V} x3r = {vc3r}, x3i = {vc3i};
        const {V} t0r = {ADD}(x0r, x2r), t0i = {ADD}(x0i, x2i);
        const {V} t1r = {SUB}(x0r, x2r), t1i = {SUB}(x0i, x2i);
        const {V} t2r = {ADD}(x1r, x3r), t2i = {ADD}(x1i, x3i);
        const {V} t3r = {SUB}(x1r, x3r), t3i = {SUB}(x1i, x3i);
        R4_ST(&rio_re[m + 0*ios], {ADD}(t0r, t2r)); R4_ST(&rio_im[m + 0*ios], {ADD}(t0i, t2i));
        R4_ST(&rio_re[m + 2*ios], {SUB}(t0r, t2r)); R4_ST(&rio_im[m + 2*ios], {SUB}(t0i, t2i));
        R4_ST(&rio_re[m + 1*ios], {r1_e}); R4_ST(&rio_im[m + 1*ios], {i1_e});
        R4_ST(&rio_re[m + 3*ios], {r3_e}); R4_ST(&rio_im[m + 3*ios], {i3_e});
    }}
}}''')

            # ── SIMD t1 DIT log1: load w1 and w2, derive w3 = w1*w2 ──
            # Saves 2 twiddle loads per iteration, costs 1 cmul (4 FMA).
            # Middle ground between flat (6 twiddle loads) and log3 (2 loads).
            # Shallower dep chain than log3: {w1, w2} → w3 is 2-deep
            # (vs log3's w1 → w2 → w3 which is 3-deep).
            # May win on chips/compilers where log3's serial chain hurts.
            parts.append(f'''
{I['target']}
static inline void
radix4_t1_dit_log1_{direction}_{isa_name}(
    double * __restrict__ rio_re, double * __restrict__ rio_im,
    const double * __restrict__ W_re, const double * __restrict__ W_im,
    size_t ios, size_t me)
{{
    /* SIMD t1 DIT log1: derive w3 = w1 * w2 (2-deep dep chain). */
    for (size_t m = 0; m < me; m += {VL}) {{
        {V} x0r = R4_LD(&rio_re[m + 0*ios]), x0i = R4_LD(&rio_im[m + 0*ios]);
        {V} r1r = R4_LD(&rio_re[m + 1*ios]), r1i = R4_LD(&rio_im[m + 1*ios]);
        {V} r2r = R4_LD(&rio_re[m + 2*ios]), r2i = R4_LD(&rio_im[m + 2*ios]);
        {V} r3r = R4_LD(&rio_re[m + 3*ios]), r3i = R4_LD(&rio_im[m + 3*ios]);
        /* Load w1 and w2, derive w3 = w1 * w2. */
        const {V} w1r = R4_LD(&W_re[0*me+m]), w1i = R4_LD(&W_im[0*me+m]);
        const {V} w2r = R4_LD(&W_re[1*me+m]), w2i = R4_LD(&W_im[1*me+m]);
        const {V} w3r = {FNMA}(w1i, w2i, {MUL}(w1r, w2r));
        const {V} w3i = {FMA}(w1r, w2i, {MUL}(w1i, w2r));
        const {V} x1r = {vc1r}, x1i = {vc1i};
        const {V} x2r = {vc2r}, x2i = {vc2i};
        const {V} x3r = {vc3r}, x3i = {vc3i};
        const {V} t0r = {ADD}(x0r, x2r), t0i = {ADD}(x0i, x2i);
        const {V} t1r = {SUB}(x0r, x2r), t1i = {SUB}(x0i, x2i);
        const {V} t2r = {ADD}(x1r, x3r), t2i = {ADD}(x1i, x3i);
        const {V} t3r = {SUB}(x1r, x3r), t3i = {SUB}(x1i, x3i);
        R4_ST(&rio_re[m + 0*ios], {ADD}(t0r, t2r)); R4_ST(&rio_im[m + 0*ios], {ADD}(t0i, t2i));
        R4_ST(&rio_re[m + 2*ios], {SUB}(t0r, t2r)); R4_ST(&rio_im[m + 2*ios], {SUB}(t0i, t2i));
        R4_ST(&rio_re[m + 1*ios], {r1_e}); R4_ST(&rio_im[m + 1*ios], {i1_e});
        R4_ST(&rio_re[m + 3*ios], {r3_e}); R4_ST(&rio_im[m + 3*ios], {i3_e});
    }}
}}''')

            # ── SIMD t1s DIT: scalar-broadcast twiddles (hoisted out of m-loop) ──
            # W_re, W_im are (R-1)=3 SCALARS, not (R-1)*me arrays. The planner
            # populates a per-stage scalar twiddle table (tw_scalar_re) and
            # arranges K-blocked execution so all me iterations within a call
            # share the same leg-twiddle. We hoist all 3 broadcasts before the
            # m-loop — fits trivially in the register budget at R=4 (6 SIMD
            # regs for re/im pairs, leaves 10 YMM / 26 ZMM for butterfly state).
            # Expected win: consumer chips at small me where twiddle load-port
            # pressure dominates (cf. R=16 Raptor Lake: t1s won 69% of regions).
            if isa_name == 'avx2':
                BCAST_RE = lambda idx: f'_mm256_broadcast_sd(&W_re[{idx}])'
                BCAST_IM = lambda idx: f'_mm256_broadcast_sd(&W_im[{idx}])'
            else:  # avx512
                BCAST_RE = lambda idx: f'_mm512_set1_pd(W_re[{idx}])'
                BCAST_IM = lambda idx: f'_mm512_set1_pd(W_im[{idx}])'
            parts.append(f'''
{I['target']}
static inline void
radix4_t1s_dit_{direction}_{isa_name}(
    double * __restrict__ rio_re, double * __restrict__ rio_im,
    const double * __restrict__ W_re, const double * __restrict__ W_im,
    size_t ios, size_t me)
{{
    /* SIMD t1s DIT: broadcast W_re[0..2], W_im[0..2] once before the loop. */
    const {V} w1r = {BCAST_RE(0)}, w1i = {BCAST_IM(0)};
    const {V} w2r = {BCAST_RE(1)}, w2i = {BCAST_IM(1)};
    const {V} w3r = {BCAST_RE(2)}, w3i = {BCAST_IM(2)};
    for (size_t m = 0; m < me; m += {VL}) {{
        {V} x0r = R4_LD(&rio_re[m + 0*ios]), x0i = R4_LD(&rio_im[m + 0*ios]);
        {V} r1r = R4_LD(&rio_re[m + 1*ios]), r1i = R4_LD(&rio_im[m + 1*ios]);
        {V} r2r = R4_LD(&rio_re[m + 2*ios]), r2i = R4_LD(&rio_im[m + 2*ios]);
        {V} r3r = R4_LD(&rio_re[m + 3*ios]), r3i = R4_LD(&rio_im[m + 3*ios]);
        const {V} x1r = {vc1r}, x1i = {vc1i};
        const {V} x2r = {vc2r}, x2i = {vc2i};
        const {V} x3r = {vc3r}, x3i = {vc3i};
        const {V} t0r = {ADD}(x0r, x2r), t0i = {ADD}(x0i, x2i);
        const {V} t1r = {SUB}(x0r, x2r), t1i = {SUB}(x0i, x2i);
        const {V} t2r = {ADD}(x1r, x3r), t2i = {ADD}(x1i, x3i);
        const {V} t3r = {SUB}(x1r, x3r), t3i = {SUB}(x1i, x3i);
        R4_ST(&rio_re[m + 0*ios], {ADD}(t0r, t2r)); R4_ST(&rio_im[m + 0*ios], {ADD}(t0i, t2i));
        R4_ST(&rio_re[m + 2*ios], {SUB}(t0r, t2r)); R4_ST(&rio_im[m + 2*ios], {SUB}(t0i, t2i));
        R4_ST(&rio_re[m + 1*ios], {r1_e}); R4_ST(&rio_im[m + 1*ios], {i1_e});
        R4_ST(&rio_re[m + 3*ios], {r3_e}); R4_ST(&rio_im[m + 3*ios], {i3_e});
    }}
}}''')

            # ── SIMD t1 DIT u2: unroll m-loop by 2 ──
            # Processes 2x k_step butterflies per outer iteration. The compiler
            # interleaves the two blocks across both FMA ports, hiding latency
            # in dependency chains. Tail handles odd (me/VL) trip counts.
            # Expected win: large-me regimes where iteration latency dominates.
            parts.append(f'''
{I['target']}
static inline void
radix4_t1_dit_u2_{direction}_{isa_name}(
    double * __restrict__ rio_re, double * __restrict__ rio_im,
    const double * __restrict__ W_re, const double * __restrict__ W_im,
    size_t ios, size_t me)
{{
    /* SIMD t1 DIT u2: m-loop unrolled 2x for ILP. */
    const size_t me_unroll = (me / {VL*2}) * {VL*2};
    size_t m = 0;
    for (; m < me_unroll; m += {VL*2}) {{
        {{ /* Block A: m */
            {V} x0r = R4_LD(&rio_re[m + 0*ios]), x0i = R4_LD(&rio_im[m + 0*ios]);
            {V} r1r = R4_LD(&rio_re[m + 1*ios]), r1i = R4_LD(&rio_im[m + 1*ios]);
            {V} r2r = R4_LD(&rio_re[m + 2*ios]), r2i = R4_LD(&rio_im[m + 2*ios]);
            {V} r3r = R4_LD(&rio_re[m + 3*ios]), r3i = R4_LD(&rio_im[m + 3*ios]);
            const {V} w1r = R4_LD(&W_re[0*me+m]), w1i = R4_LD(&W_im[0*me+m]);
            const {V} w2r = R4_LD(&W_re[1*me+m]), w2i = R4_LD(&W_im[1*me+m]);
            const {V} w3r = R4_LD(&W_re[2*me+m]), w3i = R4_LD(&W_im[2*me+m]);
            const {V} x1r = {vc1r}, x1i = {vc1i};
            const {V} x2r = {vc2r}, x2i = {vc2i};
            const {V} x3r = {vc3r}, x3i = {vc3i};
            const {V} t0r = {ADD}(x0r, x2r), t0i = {ADD}(x0i, x2i);
            const {V} t1r = {SUB}(x0r, x2r), t1i = {SUB}(x0i, x2i);
            const {V} t2r = {ADD}(x1r, x3r), t2i = {ADD}(x1i, x3i);
            const {V} t3r = {SUB}(x1r, x3r), t3i = {SUB}(x1i, x3i);
            R4_ST(&rio_re[m + 0*ios], {ADD}(t0r, t2r)); R4_ST(&rio_im[m + 0*ios], {ADD}(t0i, t2i));
            R4_ST(&rio_re[m + 2*ios], {SUB}(t0r, t2r)); R4_ST(&rio_im[m + 2*ios], {SUB}(t0i, t2i));
            R4_ST(&rio_re[m + 1*ios], {r1_e}); R4_ST(&rio_im[m + 1*ios], {i1_e});
            R4_ST(&rio_re[m + 3*ios], {r3_e}); R4_ST(&rio_im[m + 3*ios], {i3_e});
        }}
        {{ /* Block B: m+{VL} */
            const size_t mb = m + {VL};
            {V} x0r = R4_LD(&rio_re[mb + 0*ios]), x0i = R4_LD(&rio_im[mb + 0*ios]);
            {V} r1r = R4_LD(&rio_re[mb + 1*ios]), r1i = R4_LD(&rio_im[mb + 1*ios]);
            {V} r2r = R4_LD(&rio_re[mb + 2*ios]), r2i = R4_LD(&rio_im[mb + 2*ios]);
            {V} r3r = R4_LD(&rio_re[mb + 3*ios]), r3i = R4_LD(&rio_im[mb + 3*ios]);
            const {V} w1r = R4_LD(&W_re[0*me+mb]), w1i = R4_LD(&W_im[0*me+mb]);
            const {V} w2r = R4_LD(&W_re[1*me+mb]), w2i = R4_LD(&W_im[1*me+mb]);
            const {V} w3r = R4_LD(&W_re[2*me+mb]), w3i = R4_LD(&W_im[2*me+mb]);
            const {V} x1r = {vc1r}, x1i = {vc1i};
            const {V} x2r = {vc2r}, x2i = {vc2i};
            const {V} x3r = {vc3r}, x3i = {vc3i};
            const {V} t0r = {ADD}(x0r, x2r), t0i = {ADD}(x0i, x2i);
            const {V} t1r = {SUB}(x0r, x2r), t1i = {SUB}(x0i, x2i);
            const {V} t2r = {ADD}(x1r, x3r), t2i = {ADD}(x1i, x3i);
            const {V} t3r = {SUB}(x1r, x3r), t3i = {SUB}(x1i, x3i);
            R4_ST(&rio_re[mb + 0*ios], {ADD}(t0r, t2r)); R4_ST(&rio_im[mb + 0*ios], {ADD}(t0i, t2i));
            R4_ST(&rio_re[mb + 2*ios], {SUB}(t0r, t2r)); R4_ST(&rio_im[mb + 2*ios], {SUB}(t0i, t2i));
            R4_ST(&rio_re[mb + 1*ios], {r1_e}); R4_ST(&rio_im[mb + 1*ios], {i1_e});
            R4_ST(&rio_re[mb + 3*ios], {r3_e}); R4_ST(&rio_im[mb + 3*ios], {i3_e});
        }}
    }}
    /* Tail: handle the odd remainder (me not a multiple of 2*VL) */
    for (; m < me; m += {VL}) {{
        {V} x0r = R4_LD(&rio_re[m + 0*ios]), x0i = R4_LD(&rio_im[m + 0*ios]);
        {V} r1r = R4_LD(&rio_re[m + 1*ios]), r1i = R4_LD(&rio_im[m + 1*ios]);
        {V} r2r = R4_LD(&rio_re[m + 2*ios]), r2i = R4_LD(&rio_im[m + 2*ios]);
        {V} r3r = R4_LD(&rio_re[m + 3*ios]), r3i = R4_LD(&rio_im[m + 3*ios]);
        const {V} w1r = R4_LD(&W_re[0*me+m]), w1i = R4_LD(&W_im[0*me+m]);
        const {V} w2r = R4_LD(&W_re[1*me+m]), w2i = R4_LD(&W_im[1*me+m]);
        const {V} w3r = R4_LD(&W_re[2*me+m]), w3i = R4_LD(&W_im[2*me+m]);
        const {V} x1r = {vc1r}, x1i = {vc1i};
        const {V} x2r = {vc2r}, x2i = {vc2i};
        const {V} x3r = {vc3r}, x3i = {vc3i};
        const {V} t0r = {ADD}(x0r, x2r), t0i = {ADD}(x0i, x2i);
        const {V} t1r = {SUB}(x0r, x2r), t1i = {SUB}(x0i, x2i);
        const {V} t2r = {ADD}(x1r, x3r), t2i = {ADD}(x1i, x3i);
        const {V} t3r = {SUB}(x1r, x3r), t3i = {SUB}(x1i, x3i);
        R4_ST(&rio_re[m + 0*ios], {ADD}(t0r, t2r)); R4_ST(&rio_im[m + 0*ios], {ADD}(t0i, t2i));
        R4_ST(&rio_re[m + 2*ios], {SUB}(t0r, t2r)); R4_ST(&rio_im[m + 2*ios], {SUB}(t0i, t2i));
        R4_ST(&rio_re[m + 1*ios], {r1_e}); R4_ST(&rio_im[m + 1*ios], {i1_e});
        R4_ST(&rio_re[m + 3*ios], {r3_e}); R4_ST(&rio_im[m + 3*ios], {i3_e});
    }}
}}''')

            # ── SIMD t1 DIF: in-place butterfly, then post-twiddle outputs ──
            vy1r, vy1i = cmul_v(r1_e, i1_e, 'w1r', 'w1i')
            vy2r, vy2i = cmul_v(f'{SUB}(t0r, t2r)', f'{SUB}(t0i, t2i)', 'w2r', 'w2i')
            vy3r, vy3i = cmul_v(r3_e, i3_e, 'w3r', 'w3i')
            parts.append(f'''
{I['target']}
static inline void
radix4_t1_dif_{direction}_{isa_name}(
    double * __restrict__ rio_re, double * __restrict__ rio_im,
    const double * __restrict__ W_re, const double * __restrict__ W_im,
    size_t ios, size_t me)
{{
    /* SIMD t1 DIF: ms=1, mb=0 implied. In-place butterfly + post-twiddle. */
    for (size_t m = 0; m < me; m += {VL}) {{
        const {V} x0r = R4_LD(&rio_re[m + 0*ios]), x0i = R4_LD(&rio_im[m + 0*ios]);
        const {V} x1r = R4_LD(&rio_re[m + 1*ios]), x1i = R4_LD(&rio_im[m + 1*ios]);
        const {V} x2r = R4_LD(&rio_re[m + 2*ios]), x2i = R4_LD(&rio_im[m + 2*ios]);
        const {V} x3r = R4_LD(&rio_re[m + 3*ios]), x3i = R4_LD(&rio_im[m + 3*ios]);
        const {V} t0r = {ADD}(x0r, x2r), t0i = {ADD}(x0i, x2i);
        const {V} t1r = {SUB}(x0r, x2r), t1i = {SUB}(x0i, x2i);
        const {V} t2r = {ADD}(x1r, x3r), t2i = {ADD}(x1i, x3i);
        const {V} t3r = {SUB}(x1r, x3r), t3i = {SUB}(x1i, x3i);
        const {V} w1r = R4_LD(&W_re[0*me+m]), w1i = R4_LD(&W_im[0*me+m]);
        const {V} w2r = R4_LD(&W_re[1*me+m]), w2i = R4_LD(&W_im[1*me+m]);
        const {V} w3r = R4_LD(&W_re[2*me+m]), w3i = R4_LD(&W_im[2*me+m]);
        R4_ST(&rio_re[m + 0*ios], {ADD}(t0r, t2r)); R4_ST(&rio_im[m + 0*ios], {ADD}(t0i, t2i));
        R4_ST(&rio_re[m + 1*ios], {vy1r}); R4_ST(&rio_im[m + 1*ios], {vy1i});
        R4_ST(&rio_re[m + 2*ios], {vy2r}); R4_ST(&rio_im[m + 2*ios], {vy2i});
        R4_ST(&rio_re[m + 3*ios], {vy3r}); R4_ST(&rio_im[m + 3*ios], {vy3i});
    }}
}}''')

            # ── SIMD t1_oop DIT: out-of-place twiddle + butterfly ──
            vc1r_oop, vc1i_oop = cmul_v('r1r','r1i','w1r','w1i')
            vc2r_oop, vc2i_oop = cmul_v('r2r','r2i','w2r','w2i')
            vc3r_oop, vc3i_oop = cmul_v('r3r','r3i','w3r','w3i')
            parts.append(f'''
{I['target']}
static inline void
radix4_t1_oop_dit_{direction}_{isa_name}(
    const double * __restrict__ in_re, const double * __restrict__ in_im,
    double * __restrict__ out_re, double * __restrict__ out_im,
    const double * __restrict__ W_re, const double * __restrict__ W_im,
    size_t is, size_t os, size_t me)
{{
    /* SIMD t1_oop DIT: out-of-place twiddle + butterfly. */
    for (size_t m = 0; m < me; m += {VL}) {{
        {V} x0r = R4_LD(&in_re[m + 0*is]), x0i = R4_LD(&in_im[m + 0*is]);
        {V} r1r = R4_LD(&in_re[m + 1*is]), r1i = R4_LD(&in_im[m + 1*is]);
        {V} r2r = R4_LD(&in_re[m + 2*is]), r2i = R4_LD(&in_im[m + 2*is]);
        {V} r3r = R4_LD(&in_re[m + 3*is]), r3i = R4_LD(&in_im[m + 3*is]);
        const {V} w1r = R4_LD(&W_re[0*me+m]), w1i = R4_LD(&W_im[0*me+m]);
        const {V} w2r = R4_LD(&W_re[1*me+m]), w2i = R4_LD(&W_im[1*me+m]);
        const {V} w3r = R4_LD(&W_re[2*me+m]), w3i = R4_LD(&W_im[2*me+m]);
        const {V} x1r = {vc1r_oop}, x1i = {vc1i_oop};
        const {V} x2r = {vc2r_oop}, x2i = {vc2i_oop};
        const {V} x3r = {vc3r_oop}, x3i = {vc3i_oop};
        const {V} t0r = {ADD}(x0r, x2r), t0i = {ADD}(x0i, x2i);
        const {V} t1r = {SUB}(x0r, x2r), t1i = {SUB}(x0i, x2i);
        const {V} t2r = {ADD}(x1r, x3r), t2i = {ADD}(x1i, x3i);
        const {V} t3r = {SUB}(x1r, x3r), t3i = {SUB}(x1i, x3i);
        R4_ST(&out_re[m + 0*os], {ADD}(t0r, t2r)); R4_ST(&out_im[m + 0*os], {ADD}(t0i, t2i));
        R4_ST(&out_re[m + 2*os], {SUB}(t0r, t2r)); R4_ST(&out_im[m + 2*os], {SUB}(t0i, t2i));
        R4_ST(&out_re[m + 1*os], {r1_e}); R4_ST(&out_im[m + 1*os], {i1_e});
        R4_ST(&out_re[m + 3*os], {r3_e}); R4_ST(&out_im[m + 3*os], {i3_e});
    }}
}}''')

            # ── SIMD n1_scaled: n1 with output *= scale ──
            MUL = f'{p}_mul_pd'
            SET1 = f'{p}_set1_pd'
            parts.append(f'''
{I['target']}
static inline void
radix4_n1_scaled_{direction}_{isa_name}(
    const double * __restrict__ in_re, const double * __restrict__ in_im,
    double * __restrict__ out_re, double * __restrict__ out_im,
    size_t is, size_t os, size_t vl, double scale)
{{
    const {V} vscale = {SET1}(scale);
    for (size_t k = 0; k < vl; k += {VL}) {{
        const {V} x0r = R4_LD(&in_re[0*is+k]), x0i = R4_LD(&in_im[0*is+k]);
        const {V} x1r = R4_LD(&in_re[1*is+k]), x1i = R4_LD(&in_im[1*is+k]);
        const {V} x2r = R4_LD(&in_re[2*is+k]), x2i = R4_LD(&in_im[2*is+k]);
        const {V} x3r = R4_LD(&in_re[3*is+k]), x3i = R4_LD(&in_im[3*is+k]);
        const {V} t0r = {ADD}(x0r, x2r), t0i = {ADD}(x0i, x2i);
        const {V} t1r = {SUB}(x0r, x2r), t1i = {SUB}(x0i, x2i);
        const {V} t2r = {ADD}(x1r, x3r), t2i = {ADD}(x1i, x3i);
        const {V} t3r = {SUB}(x1r, x3r), t3i = {SUB}(x1i, x3i);
        R4_ST(&out_re[0*os+k], {MUL}(vscale, {ADD}(t0r, t2r))); R4_ST(&out_im[0*os+k], {MUL}(vscale, {ADD}(t0i, t2i)));
        R4_ST(&out_re[2*os+k], {MUL}(vscale, {SUB}(t0r, t2r))); R4_ST(&out_im[2*os+k], {MUL}(vscale, {SUB}(t0i, t2i)));
        R4_ST(&out_re[1*os+k], {MUL}(vscale, {r1_e})); R4_ST(&out_im[1*os+k], {MUL}(vscale, {i1_e}));
        R4_ST(&out_re[3*os+k], {MUL}(vscale, {r3_e})); R4_ST(&out_im[3*os+k], {MUL}(vscale, {i3_e}));
    }}
}}''')

    if not is_scalar:
        parts.append('\n#undef R4_LD\n#undef R4_ST')

    parts.append(f'\n{I["guard_end"]}')
    parts.append(f'#endif /* {guard} */')
    return '\n'.join(parts)


# ═══════════════════════════════════════════════════════════════
# Variant metadata for bench/dispatcher infrastructure.
#
# Maps variant_id → (function_base, protocol). Protocol drives how the
# bench harness allocates the twiddle buffer:
#   flat: W_re/W_im has (R-1)*me doubles — read per m
#   log3: W_re/W_im has 1*me doubles — read per m, derive w2, w3
#   t1s:  W_re/W_im has (R-1) doubles  — broadcast scalars, same for all m
#
# function_base has {isa}/{direction} appended to form the C symbol.
# ═══════════════════════════════════════════════════════════════

VARIANTS = {
    # flat protocol — share dispatcher 't1_dit'
    'ct_t1_dit':        ('radix4_t1_dit',        'flat', 't1_dit'),
    'ct_t1_dit_u2':     ('radix4_t1_dit_u2',     'flat', 't1_dit'),
    'ct_t1_dit_log1':   ('radix4_t1_dit_log1',   'flat', 't1_dit'),
    # log3 protocol — own dispatcher
    'ct_t1_dit_log3':   ('radix4_t1_dit_log3',   'log3', 't1_dit_log3'),
    # t1s protocol — own dispatcher
    'ct_t1s_dit':       ('radix4_t1s_dit',       't1s',  't1s_dit'),
}

def function_name(variant_id, isa, direction):
    """Return the C symbol for (variant, isa, direction).
    E.g. ('ct_t1_dit_u2', 'avx2', 'fwd') → 'radix4_t1_dit_u2_fwd_avx2'"""
    if variant_id not in VARIANTS:
        raise KeyError(f"unknown variant {variant_id}; known: {list(VARIANTS)}")
    base, _, _ = VARIANTS[variant_id]
    return f'{base}_{direction}_{isa}'

def protocol(variant_id):
    """Return the twiddle-table layout: 'flat' | 'log3' | 't1s'."""
    return VARIANTS[variant_id][1]

def dispatcher(variant_id):
    """Return the dispatcher slot this variant fills. Variants sharing
    a dispatcher slot compute the same mathematical function and are
    mutually exchangeable; variants in different slots (e.g. DIT vs DIF)
    are NOT and must be chosen at plan-construction time."""
    return VARIANTS[variant_id][2]


if __name__ == '__main__':
    import argparse
    # Backward-compat: bare positional "avx2" | "avx512" | "scalar" still works.
    if len(sys.argv) == 2 and sys.argv[1] in ISA:
        print(gen_file(sys.argv[1]))
        sys.exit(0)

    ap = argparse.ArgumentParser(
        description='Generate DFT-4 codelets for VectorFFT.')
    ap.add_argument('--isa', choices=list(ISA),
                    help='ISA target (avx2 | avx512 | scalar)')
    ap.add_argument('--variant', choices=list(VARIANTS),
                    help='Variant ID — when supplied with --list-name, '
                         'prints the C symbol for (variant, isa, direction) '
                         'and exits without generating code.')
    ap.add_argument('--direction', choices=['fwd', 'bwd'], default='fwd',
                    help='Direction for --list-name (default: fwd).')
    ap.add_argument('--list-name', action='store_true',
                    help='Print the C symbol for --variant/--isa/--direction.')
    ap.add_argument('--list-variants', action='store_true',
                    help='Print all variant IDs (one per line) and exit.')
    ap.add_argument('--list-protocol', action='store_true',
                    help='Print the protocol (flat/log3/t1s) for --variant.')
    args = ap.parse_args()

    if args.list_variants:
        for v in VARIANTS:
            print(v)
        sys.exit(0)

    if args.list_protocol:
        if not args.variant:
            print('--list-protocol requires --variant', file=sys.stderr)
            sys.exit(2)
        print(protocol(args.variant))
        sys.exit(0)

    if args.list_name:
        if not args.variant or not args.isa:
            print('--list-name requires --variant and --isa', file=sys.stderr)
            sys.exit(2)
        print(function_name(args.variant, args.isa, args.direction))
        sys.exit(0)

    if not args.isa:
        ap.print_help(sys.stderr)
        sys.exit(2)
    print(gen_file(args.isa))
