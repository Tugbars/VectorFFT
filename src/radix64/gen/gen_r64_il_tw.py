#!/usr/bin/env python3
"""
gen_r64_il_tw.py — DFT-64 interleaved (IL) twiddled codelet (8×8 CT + log3), multi-ISA.

Targets: avx2, avx512
Emits: DIT fwd/bwd + DIF fwd/bwd in one header file.

IL layout: data in [re0,im0,re1,im1,...], twiddles stay split.
Deinterleave on load, reinterleave on store. Internal computation in split.

Usage:
  python3 gen_r64_il_tw.py avx2   > fft_radix64_avx2_il_tw.h
  python3 gen_r64_il_tw.py avx512 > fft_radix64_avx512_il_tw.h
"""
import sys, os

# Import the split tw generator for all the shared logic
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_r64_tw import (ISA, Emitter, emit_itw_static_arrays,
                         emit_log3_derivation, emit_r8,
                         emit_itw_apply, emit_ext_twiddle)


# ═══════════════════════════════════════
# ISA-specific IL load/store patterns
# ═══════════════════════════════════════

def emit_il_load(isa, em, vr, vi, idx_expr):
    """Emit deinterleave load: il[re0,im0,re1,im1,...] → vr, vi."""
    T = isa.T
    if isa.name == 'avx512':
        em.o(f'{{ {T} _lo = _mm512_load_pd(&in[2*({idx_expr})]); '
             f'{T} _hi = _mm512_load_pd(&in[2*({idx_expr})+8]);')
        em.o(f'  {vr} = _mm512_permutexvar_pd(_mm512_set_epi64(7,5,3,1,6,4,2,0), _mm512_unpacklo_pd(_lo,_hi));')
        em.o(f'  {vi} = _mm512_permutexvar_pd(_mm512_set_epi64(7,5,3,1,6,4,2,0), _mm512_unpackhi_pd(_lo,_hi)); }}')
    else:  # avx2
        em.o(f'{{ {T} _lo = _mm256_load_pd(&in[2*({idx_expr})]); '
             f'{T} _hi = _mm256_load_pd(&in[2*({idx_expr})+4]);')
        em.o(f'  {vr} = _mm256_permute4x64_pd(_mm256_unpacklo_pd(_lo,_hi), 0xD8);')
        em.o(f'  {vi} = _mm256_permute4x64_pd(_mm256_unpackhi_pd(_lo,_hi), 0xD8); }}')


def emit_il_store(isa, em, vr, vi, idx_expr):
    """Emit reinterleave store: vr, vi → il[re0,im0,re1,im1,...]."""
    T = isa.T
    if isa.name == 'avx512':
        em.o(f'{{ {T} _rp = _mm512_permutexvar_pd(_mm512_set_epi64(7,3,6,2,5,1,4,0), {vr});')
        em.o(f'  {T} _ip = _mm512_permutexvar_pd(_mm512_set_epi64(7,3,6,2,5,1,4,0), {vi});')
        em.o(f'  _mm512_store_pd(&out[2*({idx_expr})], _mm512_unpacklo_pd(_rp,_ip));')
        em.o(f'  _mm512_store_pd(&out[2*({idx_expr})+8], _mm512_unpackhi_pd(_rp,_ip)); }}')
    else:  # avx2
        em.o(f'{{ {T} _rp = _mm256_permute4x64_pd({vr}, 0xD8);')
        em.o(f'  {T} _ip = _mm256_permute4x64_pd({vi}, 0xD8);')
        em.o(f'  _mm256_store_pd(&out[2*({idx_expr})], _mm256_unpacklo_pd(_rp,_ip));')
        em.o(f'  _mm256_store_pd(&out[2*({idx_expr})+4], _mm256_unpackhi_pd(_rp,_ip)); }}')


# ═══════════════════════════════════════
# DIT IL generator
# ═══════════════════════════════════════

def gen_dit_il_tw(isa, direction):
    fwd = direction == 'fwd'
    em = Emitter()
    T, C = isa.T, isa.C

    em.L.append(isa.attr)
    em.L.append(f'static void')
    em.L.append(f'radix64_tw_flat_dit_kernel_{direction}_il_{isa.name}(')
    em.L.append(f'    const double * __restrict__ in,')
    em.L.append(f'    double * __restrict__ out,')
    em.L.append(f'    const double * __restrict__ tw_re, const double * __restrict__ tw_im,')
    em.L.append(f'    size_t K)')
    em.L.append(f'{{')
    em.ind = 1

    em.o(isa.sign_mask)
    em.o(f'const {T} vc = {isa.set1("0.707106781186547524400844362104849039284835938")};')
    em.o(f'const {T} vnc = {isa.set1("-0.707106781186547524400844362104849039284835938")};')
    em.o(f'__attribute__((aligned({isa.align}))) double sp_re[64*{C}], sp_im[64*{C}];')
    if isa.name == 'avx2':
        em.o(f'__attribute__((aligned({isa.align}))) double bfr[4*{C}], bfi[4*{C}];')
    em.o(f'{T} x0r,x0i,x1r,x1i,x2r,x2i,x3r,x3i,x4r,x4i,x5r,x5i,x6r,x6i,x7r,x7i;')
    em.b()

    em.o(f'for (size_t k = 0; k < K; k += {C}) {{')
    em.ind += 1

    emit_log3_derivation(isa, em)

    xr = [f'x{i}r' for i in range(8)]
    xi = [f'x{i}i' for i in range(8)]

    for n2 in range(8):
        em.c(f'Sub-FFT n2={n2}')
        for n1 in range(8):
            n = 8*n1 + n2
            emit_il_load(isa, em, xr[n1], xi[n1], f'{n}*K+k')
            emit_ext_twiddle(isa, em, xr[n1], xi[n1], n, fwd)
        em.b()
        emit_r8(isa, em, fwd, xr, xi, xr, xi)
        em.b()
        for k1 in range(8):
            slot = n2 * 8 + k1
            em.o(f'{isa.store(f"&sp_re[{slot}*{C}]", xr[k1])}; {isa.store(f"&sp_im[{slot}*{C}]", xi[k1])};')
        em.b()

    for k1 in range(8):
        em.c(f'Column k1={k1}')
        for n2 in range(8):
            slot = n2 * 8 + k1
            em.o(f'{xr[n2]} = {isa.load(f"&sp_re[{slot}*{C}]")}; {xi[n2]} = {isa.load(f"&sp_im[{slot}*{C}]")};')
        if k1 > 0:
            for n2 in range(1, 8):
                emit_itw_apply(isa, em, xr[n2], xi[n2], (n2*k1)%64, fwd)
        em.b()
        emit_r8(isa, em, fwd, xr, xi, xr, xi)
        em.b()
        for k2 in range(8):
            m = k1 + 8 * k2
            emit_il_store(isa, em, xr[k2], xi[k2], f'{m}*K+k')
        em.b()

    em.ind -= 1
    em.o('}')
    em.L.append('}')
    em.L.append('')
    return em.L


# ═══════════════════════════════════════
# DIF IL generator
# ═══════════════════════════════════════

def gen_dif_il_tw(isa, direction):
    fwd = direction == 'fwd'
    em = Emitter()
    T, C = isa.T, isa.C

    em.L.append(isa.attr)
    em.L.append(f'static void')
    em.L.append(f'radix64_tw_flat_dif_kernel_{direction}_il_{isa.name}(')
    em.L.append(f'    const double * __restrict__ in,')
    em.L.append(f'    double * __restrict__ out,')
    em.L.append(f'    const double * __restrict__ tw_re, const double * __restrict__ tw_im,')
    em.L.append(f'    size_t K)')
    em.L.append(f'{{')
    em.ind = 1

    em.o(isa.sign_mask)
    em.o(f'const {T} vc = {isa.set1("0.707106781186547524400844362104849039284835938")};')
    em.o(f'const {T} vnc = {isa.set1("-0.707106781186547524400844362104849039284835938")};')
    em.o(f'__attribute__((aligned({isa.align}))) double sp_re[64*{C}], sp_im[64*{C}];')
    em.o(f'{T} x0r,x0i,x1r,x1i,x2r,x2i,x3r,x3i,x4r,x4i,x5r,x5i,x6r,x6i,x7r,x7i;')
    em.b()
    if isa.name == 'avx2':
        em.o(f'__attribute__((aligned({isa.align}))) double bfr[4*{C}], bfi[4*{C}];')

    em.o(f'for (size_t k = 0; k < K; k += {C}) {{')
    em.ind += 1

    emit_log3_derivation(isa, em)

    xr = [f'x{i}r' for i in range(8)]
    xi = [f'x{i}i' for i in range(8)]

    for n2 in range(8):
        em.c(f'Sub-FFT n2={n2} (DIF: no external twiddle on inputs)')
        for n1 in range(8):
            n = 8*n1 + n2
            emit_il_load(isa, em, xr[n1], xi[n1], f'{n}*K+k')
        em.b()
        emit_r8(isa, em, fwd, xr, xi, xr, xi)
        em.b()
        for k1 in range(8):
            slot = n2 * 8 + k1
            em.o(f'{isa.store(f"&sp_re[{slot}*{C}]", xr[k1])}; {isa.store(f"&sp_im[{slot}*{C}]", xi[k1])};')
        em.b()

    for k1 in range(8):
        em.c(f'Column k1={k1}')
        for n2 in range(8):
            slot = n2 * 8 + k1
            em.o(f'{xr[n2]} = {isa.load(f"&sp_re[{slot}*{C}]")}; {xi[n2]} = {isa.load(f"&sp_im[{slot}*{C}]")};')
        if k1 > 0:
            for n2 in range(1, 8):
                emit_itw_apply(isa, em, xr[n2], xi[n2], (n2*k1)%64, fwd)
        em.b()
        emit_r8(isa, em, fwd, xr, xi, xr, xi)
        em.b()
        for k2 in range(8):
            m = k1 + 8 * k2
            emit_ext_twiddle(isa, em, xr[k2], xi[k2], m, fwd)
            emit_il_store(isa, em, xr[k2], xi[k2], f'{m}*K+k')
        em.b()

    em.ind -= 1
    em.o('}')
    em.L.append('}')
    em.L.append('')
    return em.L


# ═══════════════════════════════════════
# File generator
# ═══════════════════════════════════════

def gen_file(isa_name):
    isa = ISA(isa_name)
    ISA_U = isa_name.upper()
    guard = f'FFT_RADIX64_{ISA_U}_IL_TW_H'
    L = ['/**',
         f' * @file fft_radix64_{isa_name}_il_tw.h',
         f' * @brief DFT-64 {ISA_U} interleaved — 8x8 CT + log3 twiddles (v2)',
         f' * IL layout: data in [re0,im0,re1,im1,...], twiddles stay split.',
         f' * Deinterleave on load, reinterleave on store.',
         f' * Vector width: {isa.C} doubles, k-step: {isa.C}',
         f' * Generated by gen_r64_il_tw.py {isa_name}',
         ' */', '',
         f'#ifndef {guard}', f'#define {guard}',
         '#include <stddef.h>', '#include <immintrin.h>', '']

    L.extend(emit_itw_static_arrays())

    for d in ('fwd', 'bwd'):
        L.extend(gen_dit_il_tw(isa, d))
        L.append('')
    for d in ('fwd', 'bwd'):
        L.extend(gen_dif_il_tw(isa, d))
        L.append('')

    L.append(f'#endif /* {guard} */')
    return L


if __name__ == '__main__':
    if len(sys.argv) < 2 or sys.argv[1] not in ('avx2', 'avx512'):
        print(f"Usage: {sys.argv[0]} <avx2|avx512>", file=sys.stderr)
        sys.exit(1)
    print('\n'.join(gen_file(sys.argv[1])))
