#!/usr/bin/env python3
"""
gen_r20_avx2.py — DFT-20 production codelet generator

Architecture (from bench session, March 2026):
  Phase 1: Twiddle load+apply
    - Contiguous flat: 19 sequential loads from VFFT_TW_CONTIG table
    - Binary ladder:   5 base loads + W³ precompute + 14 cmul derivations
  Phase 2: Pure butterfly (genfft n1_20 scalar FMA DAG — 208 ops, proven 5-15% faster than FFTW)
  Phase 3: Output stores
    - Direct strided stores (default)
    - Buffered: write to 1.3KB stack buffer, then scatter (wins at K=256)

Dispatch: calibrated K threshold selects contiguous vs ladder.

Backward direction: pointer-swap trick (swap re<->im on input+output).

Usage: python3 gen_r20_avx2.py <fftw_src_dir> <avx2|avx512>

Credit: Butterfly DAG from FFTW genfft (Frigo & Johnson, 1999/2005).
Twiddle strategies, contiguous layout, binary ladder: VectorFFT (2026).
"""
import re, sys, os, math

R = 20

# ════════════════════════════════════════
# Parse FFTW n1_20 (NOTW) scalar FMA DAG
# ════════════════════════════════════════

def parse_n1_dag(fftw_src, radix):
    """Extract pure butterfly DAG from FFTW n1_R scalar FMA codelet."""
    path = os.path.join(fftw_src, 'dft', 'scalar', 'codelets', f'n1_{radix}.c')
    with open(path) as f:
        code = f.read()
    m = re.search(r'#if defined\(ARCH_PREFERS_FMA\).*?\n(.*?)^#else', code, re.DOTALL | re.MULTILINE)
    body = m.group(1)

    ops = []
    constants = {}
    for line in body.split('\n'):
        line = line.strip().rstrip(';').strip()
        if not line or line.startswith('//') or line.startswith('/*') or line.startswith('*'):
            continue
        if line.startswith('{') or line.startswith('}') or 'MAKE_VOLATILE' in line:
            continue
        if 'INT ' in line or 'for (v' in line or 'for (i' in line or line.startswith('E '):
            continue
        if line.startswith('DK('):
            m2 = re.match(r'DK\((\w+),\s*([^)]+)\)', line)
            if m2:
                constants[m2.group(1)] = m2.group(2).strip()
            continue
        # Stores
        m2 = re.match(r'(ro|io)\[(?:WS\(os,\s*(\d+)\)|0)\]\s*=\s*(.+)', line)
        if m2:
            arr = 're' if m2.group(1) == 'ro' else 'im'
            idx = int(m2.group(2)) if m2.group(2) else 0
            ops.append(('STORE', arr, idx, m2.group(3)))
            continue
        # Assignments
        m2 = re.match(r'(\w+)\s*=\s*(.+)', line)
        if not m2:
            continue
        tgt, expr = m2.group(1), m2.group(2)
        # Loads
        lm = re.match(r'(ri|ii)\[(?:WS\(is,\s*(\d+)\)|0)\]', expr)
        if lm:
            arr = 're' if lm.group(1) == 'ri' else 'im'
            idx = int(lm.group(2)) if lm.group(2) else 0
            ops.append(('LOAD', tgt, arr, idx))
            continue
        # FMA variants
        for opn in ('FMA', 'FNMS', 'FMS', 'FNMA'):
            fm = re.match(rf'{opn}\(([^,]+),\s*([^,]+),\s*([^)]+)\)', expr)
            if fm:
                ops.append((opn, tgt, fm.group(1).strip(), fm.group(2).strip(), fm.group(3).strip()))
                break
        else:
            bm = re.match(r'(.+?)\s*([+\-*])\s*(.+)', expr)
            if bm:
                ops.append(('BINOP', tgt, bm.group(2), bm.group(1).strip(), bm.group(3).strip()))
            else:
                ops.append(('UNKNOWN', tgt, expr))

    return ops, constants


# ════════════════════════════════════════
# Binary ladder derivation tree for R=20
# ════════════════════════════════════════

# 5 bases loaded: W^1(idx=0), W^2(idx=1), W^4(idx=3), W^8(idx=7), W^16(idx=15)
# W^3 = W^1 * W^2 (precomputed, reused for m=3,7,11,15,19)
# Derivation: depth-1 and depth-2 cmuls from bases + W^3

LADDER_BASES = {1: 0, 2: 1, 4: 3, 8: 7, 16: 15}  # W^exp → twiddle table index

LADDER_DERIVE = [
    # (target_m, method, args)
    # method='base': load from table at LADDER_BASES[exp]
    # method='cmul': W^a * W^b
    # method='cmul3': W^a * W^b * W^c (two cmuls)
    (1,  'base', (1,)),
    (2,  'base', (2,)),
    (3,  'cmul', (1, 2)),      # W^3 = W^1 * W^2 (kept live)
    (4,  'base', (4,)),
    (5,  'cmul', (1, 4)),
    (6,  'cmul', (2, 4)),
    (7,  'cmul', (3, 4)),      # uses precomputed W^3
    (8,  'base', (8,)),
    (9,  'cmul', (1, 8)),
    (10, 'cmul', (2, 8)),
    (11, 'cmul', (3, 8)),
    (12, 'cmul', (4, 8)),
    (13, 'cmul3', (1, 4, 8)),
    (14, 'cmul3', (2, 4, 8)),
    (15, 'cmul3', (3, 4, 8)),
    (16, 'base', (16,)),
    (17, 'cmul', (1, 16)),
    (18, 'cmul', (2, 16)),
    (19, 'cmul', (3, 16)),
]


# ════════════════════════════════════════
# ISA configuration
# ════════════════════════════════════════

def isa_config(isa):
    if isa == 'avx512':
        return dict(T='__m512d', C=8, P='_mm512', VL=8, align=64,
                    attr='__attribute__((target("avx512f,fma")))')
    else:  # avx2
        return dict(T='__m256d', C=4, P='_mm256', VL=4, align=32,
                    attr='__attribute__((target("avx2,fma")))')


# ════════════════════════════════════════
# Emitter
# ════════════════════════════════════════

class Emitter:
    def __init__(self):
        self.L = []
        self.ind = 0

    def o(self, s=''):
        self.L.append('    ' * self.ind + s)

    def c(self, s):
        self.o(f'/* {s} */')

    def b(self):
        self.L.append('')


# ════════════════════════════════════════
# Phase 2: Butterfly body emitter
# ════════════════════════════════════════

def emit_butterfly_body(em, cfg, ops, constants, out_target='out', out_stride='K'):
    """Emit the genfft DAG as inline SIMD code.
    Inputs:  xr_M / xi_M variables (pre-loaded+twiddled)
    Outputs: stores to out_target_re/im[idx * out_stride + k]
    """
    T, P = cfg['T'], cfg['P']
    cpfx = 'R20_'
    crename = {k: cpfx + k for k in constants}
    cnames = set(crename.values())

    def bc(a):
        a = a.strip()
        a = crename.get(a, a)
        if a in cnames:
            return f'{P}_set1_pd({a})'
        return a

    def translate_store_expr(expr):
        expr = expr.strip()
        for o2, p2 in crename.items():
            expr = expr.replace(o2, p2)
        for opn, fn in [('FMA', 'fmadd'), ('FNMS', 'fnmadd'), ('FMS', 'fmsub')]:
            m = re.match(rf'{opn}\(([^,]+),\s*([^,]+),\s*([^)]+)\)', expr)
            if m:
                return f'{P}_{fn}_pd({bc(m.group(1).strip())}, {bc(m.group(2).strip())}, {bc(m.group(3).strip())})'
        m = re.match(r'(\w+)\s*\+\s*(\w+)', expr)
        if m:
            return f'{P}_add_pd({bc(m.group(1))}, {bc(m.group(2))})'
        m = re.match(r'(\w+)\s*-\s*(\w+)', expr)
        if m:
            return f'{P}_sub_pd({bc(m.group(1))}, {bc(m.group(2))})'
        if re.match(r'^\w+$', expr):
            return expr
        return f'/* UNTRANSLATED: {expr} */'

    fn_map = {'FMA': 'fmadd', 'FNMS': 'fnmadd', 'FMS': 'fmsub', 'FNMA': 'fnmsub'}
    decl_set = set()

    def decl(v):
        if v not in decl_set:
            decl_set.add(v)
            return f'{T} {v}'
        return v

    for op in ops:
        if op[0] == 'LOAD':
            _, tgt, arr, idx = op
            # Input comes from pre-loaded xr_idx / xi_idx
            em.o(f'{decl(tgt)} = x{arr[0]}_{idx};')

        elif op[0] == 'STORE':
            _, arr, idx, expr = op
            et = translate_store_expr(expr)
            em.o(f'{P}_storeu_pd(&{out_target}_{arr}[{idx} * {out_stride} + k], {et});')

        elif op[0] in fn_map:
            _, tgt, a, b, c = op
            em.o(f'{decl(tgt)} = {P}_{fn_map[op[0]]}_pd({bc(a)}, {bc(b)}, {bc(c)});')

        elif op[0] == 'BINOP':
            _, tgt, sym, lhs, rhs = op
            fn = {'+': "add", '-': "sub", '*': "mul"}[sym]
            em.o(f'{decl(tgt)} = {P}_{fn}_pd({bc(lhs)}, {bc(rhs)});')

        elif op[0] == 'NEG':
            _, tgt, src = op
            em.o(f'{decl(tgt)} = {P}_xor_pd({src}, {P}_set1_pd(-0.0));')

        elif op[0] == 'UNKNOWN':
            em.o(f'/* UNKNOWN: {op[1]} = {op[2]} */')


# ════════════════════════════════════════
# Phase 1: Twiddle strategies
# ════════════════════════════════════════

def emit_contiguous_twiddle_load(em, cfg):
    """Emit contiguous flat twiddle load+apply for m=1..19."""
    T, P, VL = cfg['T'], cfg['P'], cfg['VL']

    em.c('Phase 1: Contiguous flat twiddle load+apply')
    em.o(f'const size_t kg = k / {VL};')
    em.o(f'const double *tr = &tw_re[kg * 19 * {VL}];')
    em.o(f'const double *ti = &tw_im[kg * 19 * {VL}];')
    em.b()
    em.o(f'{T} xr_0 = {P}_loadu_pd(&in_re[k]);')
    em.o(f'{T} xi_0 = {P}_loadu_pd(&in_im[k]);')

    for m in range(1, R):
        em.o(f'{T} xr_{m}, xi_{m};')
        em.o(f'{{ {T} dr = {P}_loadu_pd(&in_re[{m}*K+k]);')
        em.o(f'  {T} di = {P}_loadu_pd(&in_im[{m}*K+k]);')
        em.o(f'  {T} wr = {P}_load_pd(&tr[{m-1}*{VL}]);')
        em.o(f'  {T} wi = {P}_load_pd(&ti[{m-1}*{VL}]);')
        em.o(f'  xr_{m} = {P}_fmsub_pd(dr, wr, {P}_mul_pd(di, wi));')
        em.o(f'  xi_{m} = {P}_fmadd_pd(dr, wi, {P}_mul_pd(di, wr)); }}')


def emit_ladder_twiddle_load(em, cfg):
    """Emit binary ladder twiddle load+apply for m=1..19."""
    T, P, VL = cfg['T'], cfg['P'], cfg['VL']

    em.c('Phase 1: Binary ladder twiddle load+apply (5 loads + 14 cmuls)')
    em.o(f'{T} xr_0 = {P}_loadu_pd(&in_re[k]);')
    em.o(f'{T} xi_0 = {P}_loadu_pd(&in_im[k]);')
    em.b()

    # Load 5 bases
    em.c('Load 5 ladder bases')
    for exp, idx in sorted(LADDER_BASES.items()):
        em.o(f'{T} b{exp}r = {P}_loadu_pd(&tw_re[{idx}*K+k]);')
        em.o(f'{T} b{exp}i = {P}_loadu_pd(&tw_im[{idx}*K+k]);')
    em.b()

    # Precompute W^3
    em.c('Precompute W^3 = W^1 * W^2 (reused for m=3,7,11,15,19)')
    em.o(f'{T} w3r = {P}_fmsub_pd(b1r, b2r, {P}_mul_pd(b1i, b2i));')
    em.o(f'{T} w3i = {P}_fmadd_pd(b1r, b2i, {P}_mul_pd(b1i, b2r));')
    em.b()

    # Derive and apply for each m
    em.c('Derive twiddles + load inputs + apply')
    for m, method, args in LADDER_DERIVE:
        if method == 'base':
            exp = args[0]
            twr, twi = f'b{exp}r', f'b{exp}i'
        elif method == 'cmul':
            a, b = args
            ar = f'b{a}r' if a != 3 else 'w3r'
            ai = f'b{a}i' if a != 3 else 'w3i'
            br = f'b{b}r' if b != 3 else 'w3r'
            bi = f'b{b}i' if b != 3 else 'w3i'
            em.o(f'{T} tw{m}r = {P}_fmsub_pd({ar}, {br}, {P}_mul_pd({ai}, {bi}));')
            em.o(f'{T} tw{m}i = {P}_fmadd_pd({ar}, {bi}, {P}_mul_pd({ai}, {br}));')
            twr, twi = f'tw{m}r', f'tw{m}i'
        elif method == 'cmul3':
            a, b, c = args
            ar = f'b{a}r' if a != 3 else 'w3r'
            ai = f'b{a}i' if a != 3 else 'w3i'
            br, bi = f'b{b}r', f'b{b}i'
            cr, ci = f'b{c}r', f'b{c}i'
            twr = f'tw{m}r'
            twi = f'tw{m}i'
            em.o(f'{T} {twr}, {twi};')
            em.o(f'{{ {T} abr = {P}_fmsub_pd({ar}, {br}, {P}_mul_pd({ai}, {bi}));')
            em.o(f'  {T} abi = {P}_fmadd_pd({ar}, {bi}, {P}_mul_pd({ai}, {br}));')
            em.o(f'  {twr} = {P}_fmsub_pd(abr, {cr}, {P}_mul_pd(abi, {ci}));')
            em.o(f'  {twi} = {P}_fmadd_pd(abr, {ci}, {P}_mul_pd(abi, {cr})); }}')

        em.o(f'{T} xr_{m}, xi_{m};')
        em.o(f'{{ {T} dr = {P}_loadu_pd(&in_re[{m}*K+k]);')
        em.o(f'  {T} di = {P}_loadu_pd(&in_im[{m}*K+k]);')
        em.o(f'  xr_{m} = {P}_fmsub_pd(dr, {twr}, {P}_mul_pd(di, {twi}));')
        em.o(f'  xi_{m} = {P}_fmadd_pd(dr, {twi}, {P}_mul_pd(di, {twr})); }}')


# ════════════════════════════════════════
# Full kernel generators
# ════════════════════════════════════════

def gen_notw(cfg, dag_ops, constants):
    """Generate NOTW (N1) kernel — pure butterfly, no twiddles."""
    T, P, C = cfg['T'], cfg['P'], cfg['C']
    isa = 'avx2' if C == 4 else 'avx512'
    cpfx = 'R20_'

    em = Emitter()
    em.c(f'DFT-20 NOTW (N1) — genfft DAG, split format, {isa.upper()}')
    em.c(f'Pure butterfly: 208 FP ops, 4 constants, no twiddles.')
    em.c(f'Butterfly DAG: Frigo & Johnson (FFTW genfft, 1999/2005)')
    em.b()

    # Constants
    em.o(f'#ifndef FFT_R20_DAG_CONSTS')
    em.o(f'#define FFT_R20_DAG_CONSTS')
    for k, v in sorted(constants.items()):
        em.o(f'static const double {cpfx}{k} = {v};')
    em.o(f'#endif')
    em.b()

    em.o(cfg['attr'])
    em.o(f'static void radix20_n1_dag_fwd_{isa}(')
    em.o(f'    const double * __restrict__ in_re, const double * __restrict__ in_im,')
    em.o(f'    double * __restrict__ out_re, double * __restrict__ out_im,')
    em.o(f'    size_t K)')
    em.o(f'{{')
    em.ind = 1
    em.o(f'for (size_t k = 0; k < K; k += {C}) {{')
    em.ind += 1

    # Load all 20 inputs
    em.c('Load all 20 inputs (no twiddles)')
    for m in range(R):
        em.o(f'{T} xr_{m} = {P}_loadu_pd(&in_re[{m}*K+k]);')
        em.o(f'{T} xi_{m} = {P}_loadu_pd(&in_im[{m}*K+k]);')
    em.b()

    em.c('Butterfly: genfft n1_20 scalar FMA DAG (208 ops)')
    emit_butterfly_body(em, cfg, dag_ops, constants, 'out', 'K')

    em.ind -= 1
    em.o('}')
    em.ind -= 1
    em.o('}')
    em.b()

    # Backward via pointer swap
    em.c('Backward: swap re<->im on input+output (Frigo & Johnson)')
    em.o(cfg['attr'])
    em.o(f'static inline void radix20_n1_dag_bwd_{isa}(')
    em.o(f'    const double * __restrict__ in_re, const double * __restrict__ in_im,')
    em.o(f'    double * __restrict__ out_re, double * __restrict__ out_im,')
    em.o(f'    size_t K)')
    em.o(f'{{')
    em.ind = 1
    em.o(f'radix20_n1_dag_fwd_{isa}(in_im, in_re, out_im, out_re, K);')
    em.ind -= 1
    em.o('}')
    em.b()
    return em.L


def gen_tw_dit(cfg, dag_ops, constants, tw_mode):
    """Generate TW DIT kernel.
    tw_mode: 'contig' (contiguous flat) or 'ladder' (binary ladder)
    """
    T, P, C, VL = cfg['T'], cfg['P'], cfg['C'], cfg['VL']
    isa = 'avx2' if C == 4 else 'avx512'
    suffix = 'contig' if tw_mode == 'contig' else 'ladder'

    em = Emitter()
    em.c(f'DFT-20 TW DIT ({tw_mode}) — genfft DAG + {tw_mode} twiddles, {isa.upper()}')
    if tw_mode == 'contig':
        em.c(f'Twiddle layout: VFFT_TW_CONTIG — 19 entries packed per k-group of {VL}')
        em.c(f'Sequential access: 19 aligned loads from contiguous block. Perfect prefetch.')
    else:
        em.c(f'Twiddle layout: strided — 5 base loads + 14 cmul derivations')
        em.c(f'Binary ladder: W^1,W^2,W^4,W^8,W^16 → derive all 19 via depth≤2 cmuls')
    em.c(f'Butterfly DAG: Frigo & Johnson (FFTW genfft, 1999/2005)')
    em.b()

    em.o(cfg['attr'])
    em.o(f'static void radix20_tw_dag_dit_{suffix}_fwd_{isa}(')
    em.o(f'    const double * __restrict__ in_re, const double * __restrict__ in_im,')
    em.o(f'    double * __restrict__ out_re, double * __restrict__ out_im,')
    em.o(f'    const double * __restrict__ tw_re, const double * __restrict__ tw_im,')
    em.o(f'    size_t K)')
    em.o(f'{{')
    em.ind = 1
    em.o(f'for (size_t k = 0; k < K; k += {C}) {{')
    em.ind += 1

    if tw_mode == 'contig':
        emit_contiguous_twiddle_load(em, cfg)
    else:
        emit_ladder_twiddle_load(em, cfg)
    em.b()

    em.c('Phase 2: Butterfly (genfft n1_20 DAG)')
    emit_butterfly_body(em, cfg, dag_ops, constants, 'out', 'K')

    em.ind -= 1
    em.o('}')
    em.ind -= 1
    em.o('}')
    em.b()

    # Backward via pointer swap
    em.c(f'Backward ({tw_mode}): swap re<->im (Frigo & Johnson)')
    em.o(cfg['attr'])
    em.o(f'static inline void radix20_tw_dag_dit_{suffix}_bwd_{isa}(')
    em.o(f'    const double * __restrict__ in_re, const double * __restrict__ in_im,')
    em.o(f'    double * __restrict__ out_re, double * __restrict__ out_im,')
    em.o(f'    const double * __restrict__ tw_re, const double * __restrict__ tw_im,')
    em.o(f'    size_t K)')
    em.o(f'{{')
    em.ind = 1
    em.o(f'radix20_tw_dag_dit_{suffix}_fwd_{isa}(in_im, in_re, out_im, out_re, tw_re, tw_im, K);')
    em.ind -= 1
    em.o('}')
    em.b()
    return em.L


def gen_tw_dit_buffered(cfg, dag_ops, constants):
    """Generate TW DIT with contiguous twiddles + buffered output (v10 architecture)."""
    T, P, C, VL = cfg['T'], cfg['P'], cfg['C'], cfg['VL']
    isa = 'avx2' if C == 4 else 'avx512'

    em = Emitter()
    em.c(f'DFT-20 TW DIT (contiguous + buffered output) — {isa.upper()}')
    em.c(f'DAG writes to 1.3KB stack buffer, then one scatter pass.')
    em.c(f'Wins at L1-spill K values (K~256) where 20 strided write streams thrash cache.')
    em.b()

    em.o(cfg['attr'])
    em.o(f'static void radix20_tw_dag_dit_contig_buf_fwd_{isa}(')
    em.o(f'    const double * __restrict__ in_re, const double * __restrict__ in_im,')
    em.o(f'    double * __restrict__ out_re, double * __restrict__ out_im,')
    em.o(f'    const double * __restrict__ tw_re, const double * __restrict__ tw_im,')
    em.o(f'    size_t K)')
    em.o(f'{{')
    em.ind = 1
    em.o(f'__attribute__((aligned({cfg["align"]}))) double buf_re[{R * VL}];')
    em.o(f'__attribute__((aligned({cfg["align"]}))) double buf_im[{R * VL}];')
    em.b()
    em.o(f'for (size_t k = 0; k < K; k += {C}) {{')
    em.ind += 1

    emit_contiguous_twiddle_load(em, cfg)
    em.b()

    em.c('Phase 2: Butterfly → contiguous buffer (stride={VL})')
    emit_butterfly_body(em, cfg, dag_ops, constants, 'buf', str(VL))
    em.b()

    em.c('Phase 3: Scatter from buffer to strided output')
    em.o(f'for (int m = 0; m < {R}; m++) {{')
    em.ind += 1
    em.o(f'{P}_storeu_pd(&out_re[m * K + k], {P}_load_pd(&buf_re[m * {VL}]));')
    em.o(f'{P}_storeu_pd(&out_im[m * K + k], {P}_load_pd(&buf_im[m * {VL}]));')
    em.ind -= 1
    em.o('}')

    em.ind -= 1
    em.o('}')
    em.ind -= 1
    em.o('}')
    em.b()

    # Backward
    em.c('Backward (contiguous+buffered): swap re<->im')
    em.o(cfg['attr'])
    em.o(f'static inline void radix20_tw_dag_dit_contig_buf_bwd_{isa}(')
    em.o(f'    const double * __restrict__ in_re, const double * __restrict__ in_im,')
    em.o(f'    double * __restrict__ out_re, double * __restrict__ out_im,')
    em.o(f'    const double * __restrict__ tw_re, const double * __restrict__ tw_im,')
    em.o(f'    size_t K)')
    em.o(f'{{')
    em.ind = 1
    em.o(f'radix20_tw_dag_dit_contig_buf_fwd_{isa}(in_im, in_re, out_im, out_re, tw_re, tw_im, K);')
    em.ind -= 1
    em.o('}')
    em.b()
    return em.L


# ════════════════════════════════════════
# File generation
# ════════════════════════════════════════

def gen_file(fftw_src, isa):
    cfg = isa_config(isa)
    ISA = isa.upper()
    dag_ops, constants = parse_n1_dag(fftw_src, R)
    cpfx = 'R20_'

    guard = f'FFT_RADIX20_{ISA}_H'
    L = []
    L.append(f'/**')
    L.append(f' * @file fft_radix20_{isa}.h')
    L.append(f' * @brief DFT-20 {ISA} production codelets — split re/im format')
    L.append(f' *')
    L.append(f' * Butterfly: genfft DAG (208 ops, 4 constants) — proven 5-15% faster')
    L.append(f' * than FFTW n1fv_20 {ISA} in NOTW benchmarks.')
    L.append(f' *')
    L.append(f' * Twiddle strategies (benchmarked, March 2026):')
    L.append(f' *   K ≤ 32:  contiguous flat (19 sequential loads, perfect prefetch)')
    L.append(f' *   K = 256: contiguous flat + buffered output (1.3KB buffer, L1 sweet spot)')
    L.append(f' *   K ≥ 64:  binary ladder (5 loads + 14 cmuls, saves cache at high K)')
    L.append(f' *')
    L.append(f' * Credit: Butterfly DAG from FFTW genfft (Frigo & Johnson, 1999/2005).')
    L.append(f' * Twiddle strategies, contiguous layout, binary ladder: VectorFFT (2026).')
    L.append(f' *')
    L.append(f' * Generated by gen_r20_avx2.py')
    L.append(f' */')
    L.append(f'')
    L.append(f'#ifndef {guard}')
    L.append(f'#define {guard}')
    L.append(f'#include <stddef.h>')
    L.append(f'#include <immintrin.h>')
    L.append(f'')

    # Constants
    L.append(f'#ifndef FFT_R20_DAG_CONSTS')
    L.append(f'#define FFT_R20_DAG_CONSTS')
    for k, v in sorted(constants.items()):
        L.append(f'static const double {cpfx}{k} = {v};')
    L.append(f'#endif')
    L.append(f'')

    # NOTW
    L.extend(gen_notw(cfg, dag_ops, constants))

    # TW DIT contiguous
    L.extend(gen_tw_dit(cfg, dag_ops, constants, 'contig'))

    # TW DIT ladder
    L.extend(gen_tw_dit(cfg, dag_ops, constants, 'ladder'))

    # TW DIT contiguous + buffered
    L.extend(gen_tw_dit_buffered(cfg, dag_ops, constants))

    L.append(f'#endif /* {guard} */')
    return L


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: gen_r20_avx2.py <fftw_src_dir> <avx2|avx512>", file=sys.stderr)
        sys.exit(1)
    fftw_src = sys.argv[1]
    isa = sys.argv[2]
    lines = gen_file(fftw_src, isa)
    print('\n'.join(lines))
    n_unk = sum(1 for l in lines if 'UNKNOWN' in l or 'UNTRANSLATED' in l)
    print(f"/* {len(lines)} lines, {n_unk} unknowns */", file=sys.stderr)
