#!/usr/bin/env python3
"""gen_r20_n1.py — Generate R=20 NOTW (N1) codelet from FFTW n1_20 scalar FMA DAG."""
import re, sys, os

def extract_fma_body(fftw_src, radix):
    path = os.path.join(fftw_src, 'dft', 'scalar', 'codelets', f'n1_{radix}.c')
    with open(path) as f: code = f.read()
    m = re.search(r'#if defined\(ARCH_PREFERS_FMA\).*?\n(.*?)^#else', code, re.DOTALL|re.MULTILINE)
    return m.group(1)

def parse_body(body):
    ops = []; constants = {}
    for line in body.split('\n'):
        line = line.strip().rstrip(';').strip()
        if not line or line.startswith('//') or line.startswith('/*') or line.startswith('*'): continue
        if line.startswith('{') or line.startswith('}') or 'MAKE_VOLATILE' in line: continue
        if 'INT ' in line or 'for (v' in line or 'for (i' in line or line.startswith('E '): continue
        if line.startswith('DK('):
            m = re.match(r'DK\((\w+),\s*([^)]+)\)', line)
            if m: constants[m.group(1)] = m.group(2).strip()
            continue
        # Stores: ro/io
        m = re.match(r'(ro|io)\[(?:WS\(os,\s*(\d+)\)|0)\]\s*=\s*(.+)', line)
        if m:
            arr = m.group(1); idx = int(m.group(2)) if m.group(2) else 0
            ops.append(('STORE', 're' if arr=='ro' else 'im', idx, m.group(3)))
            continue
        m = re.match(r'(\w+)\s*=\s*(.+)', line)
        if not m: continue
        tgt, expr = m.group(1), m.group(2)
        # Loads: ri/ii
        lm = re.match(r'(ri|ii)\[(?:WS\(is,\s*(\d+)\)|0)\]', expr)
        if lm:
            arr = lm.group(1); idx = int(lm.group(2)) if lm.group(2) else 0
            ops.append(('LOAD', tgt, 're' if arr=='ri' else 'im', idx)); continue
        for opn in ('FMA','FNMS','FMS','FNMA'):
            fm = re.match(rf'{opn}\(([^,]+),\s*([^,]+),\s*([^)]+)\)', expr)
            if fm:
                ops.append((opn, tgt, fm.group(1).strip(), fm.group(2).strip(), fm.group(3).strip())); break
        else:
            bm = re.match(r'(.+?)\s*([+\-*])\s*(.+)', expr)
            if bm:
                ops.append(('BINOP', tgt, bm.group(2), bm.group(1).strip(), bm.group(3).strip()))
            else:
                ops.append(('UNKNOWN', tgt, expr))
    return ops, constants

def translate_store_expr(expr, cnames, P):
    expr = expr.strip()
    def bc(a):
        a = a.strip()
        return f'{P}_set1_pd({a})' if a in cnames else a
    for opn, fn in [('FMA','fmadd'),('FNMS','fnmadd'),('FMS','fmsub')]:
        m = re.match(rf'{opn}\(([^,]+),\s*([^,]+),\s*([^)]+)\)', expr)
        if m: return f'{P}_{fn}_pd({bc(m.group(1).strip())}, {bc(m.group(2).strip())}, {bc(m.group(3).strip())})'
    m = re.match(r'(\w+)\s*\+\s*(\w+)', expr)
    if m: return f'{P}_add_pd({bc(m.group(1))}, {bc(m.group(2))})'
    m = re.match(r'(\w+)\s*-\s*(\w+)', expr)
    if m: return f'{P}_sub_pd({bc(m.group(1))}, {bc(m.group(2))})'
    if re.match(r'^\w+$', expr): return expr
    return f'/* UNTRANSLATED: {expr} */'

def emit(ops, constants, radix, isa):
    R = radix
    if isa == 'avx512':
        T, C, P = '__m512d', 8, '_mm512'
        attr = '__attribute__((target("avx512f,fma")))'
    else:
        T, C, P = '__m256d', 4, '_mm256'
        attr = '__attribute__((target("avx2,fma")))'
    ISA_U = isa.upper()
    cpfx = f'R{R}N_'
    pfx_c = {cpfx+k: v for k,v in constants.items()}
    cnames = set(pfx_c.keys())
    crename = {k: cpfx+k for k in constants}
    def bc(a):
        a = a.strip(); a = crename.get(a, a)
        return f'{P}_set1_pd({a})' if a in cnames else a

    guard = f'FFT_RADIX{R}_{ISA_U}_N1_DAG_H'
    L = []
    L.append(f'/**')
    L.append(f' * @file fft_radix{R}_{isa}_n1_dag.h')
    L.append(f' * @brief DFT-{R} {ISA_U} split NOTW codelet — genfft DAG')
    L.append(f' * Generated from FFTW n1_{R}. Credit: Frigo & Johnson (1999/2005).')
    L.append(f' */'); L.append('')
    L.append(f'#ifndef {guard}'); L.append(f'#define {guard}')
    L.append(f'#include <stddef.h>'); L.append(f'#include <immintrin.h>'); L.append('')
    L.append(f'#ifndef FFT_RADIX{R}_N1_CONSTANTS'); L.append(f'#define FFT_RADIX{R}_N1_CONSTANTS')
    for n,v in sorted(pfx_c.items()):
        L.append(f'static const double {n} = {v};')
    L.append(f'#endif'); L.append('')
    L.append(attr)
    L.append(f'static void radix{R}_n1_dag_dit_fwd_{isa}(')
    L.append(f'    const double * __restrict__ in_re, const double * __restrict__ in_im,')
    L.append(f'    double * __restrict__ out_re, double * __restrict__ out_im,')
    L.append(f'    size_t K)')
    L.append(f'{{')
    L.append(f'    for (size_t k = 0; k < K; k += {C}) {{')
    ind = '        '; decl_set = set()
    def decl(v):
        if v not in decl_set: decl_set.add(v); return f'{T} {v}'
        return v
    for op in ops:
        if op[0] == 'LOAD':
            _, tgt, arr, idx = op
            L.append(f'{ind}{decl(tgt)} = {P}_load_pd(&in_{arr}[{idx}*K+k]);')
        elif op[0] == 'STORE':
            _, arr, idx, expr = op
            re_expr = expr
            for o,p in crename.items(): re_expr = re_expr.replace(o, p)
            et = translate_store_expr(re_expr, cnames, P)
            L.append(f'{ind}{P}_store_pd(&out_{arr}[{idx}*K+k], {et});')
        elif op[0] in ('FMA','FNMS','FMS','FNMA'):
            fn_map = {'FMA':'fmadd','FNMS':'fnmadd','FMS':'fmsub','FNMA':'fnmsub'}
            _, tgt, a, b, c = op
            L.append(f'{ind}{decl(tgt)} = {P}_{fn_map[op[0]]}_pd({bc(a)}, {bc(b)}, {bc(c)});')
        elif op[0] == 'BINOP':
            _, tgt, sym, lhs, rhs = op
            fn = {'+':"add",'-':"sub",'*':"mul"}[sym]
            L.append(f'{ind}{decl(tgt)} = {P}_{fn}_pd({bc(lhs)}, {bc(rhs)});')
        elif op[0] == 'NEG':
            _, tgt, src = op
            L.append(f'{ind}{decl(tgt)} = {P}_xor_pd({src}, {P}_set1_pd(-0.0));')
        elif op[0] == 'UNKNOWN':
            L.append(f'{ind}/* UNKNOWN: {op[1]} = {op[2]} */')
    L.append(f'    }}'); L.append(f'}}'); L.append('')
    # Backward via pointer swap
    L.append(f'/* Backward: swap re<->im (Frigo & Johnson) */')
    L.append(attr)
    L.append(f'static inline void radix{R}_n1_dag_dit_bwd_{isa}(')
    L.append(f'    const double * __restrict__ in_re, const double * __restrict__ in_im,')
    L.append(f'    double * __restrict__ out_re, double * __restrict__ out_im,')
    L.append(f'    size_t K)')
    L.append(f'{{')
    L.append(f'    radix{R}_n1_dag_dit_fwd_{isa}(in_im, in_re, out_im, out_re, K);')
    L.append(f'}}'); L.append('')
    L.append(f'#endif /* {guard} */')
    return L

fftw_src = sys.argv[1]; radix = int(sys.argv[2])
isa = sys.argv[3] if len(sys.argv) > 3 else 'avx2'
body = extract_fma_body(fftw_src, radix)
ops, constants = parse_body(body)
n_unk = sum(1 for o in ops if o[0] == 'UNKNOWN')
n_load = sum(1 for o in ops if o[0] == 'LOAD')
n_store = sum(1 for o in ops if o[0] == 'STORE')
print(f"/* N1 DAG: {n_load} loads, {n_store} stores, {n_unk} unknown */", file=sys.stderr)
print('\n'.join(emit(ops, constants, radix, isa)))
