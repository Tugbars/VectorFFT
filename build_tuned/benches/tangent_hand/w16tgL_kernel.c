/* w16tg — MKL's W-16 construction (tangent-Givens + deferred normalization),
 * regenerated from the VALIDATED interpretation of mkl512__w16_fwd_loop.asm
 * (w16_eval.py: full gate 1.9e-14 vs golden DFT16*diag). Same dataflow, SSA
 * locals, spills left to gcc. Frozen z ABI; table = 15 records/2-col group,
 * 8 doubles each, [c,c,c',c'][+s,-s,+s',-s'] (THEIR fold), leg = slot+1,
 * cursor +120 doubles per group. Constants baked: -tan(pi/8), -sqrt(1/2),
 * -cos(pi/8); free rotations +i via [-0,0] mask. */
#include <immintrin.h>
#include <stddef.h>

static const __m256d _MSK  = { -0.0, 0.0, -0.0, 0.0 };
static const __m256d _TAN8 = { -0.41421356237309503, -0.41421356237309503, -0.41421356237309503, -0.41421356237309503 };
static const __m256d _R2   = { -0.7071067811865476, -0.7071067811865476, -0.7071067811865476, -0.7071067811865476 };
static const __m256d _CP8  = { -0.9238795325112867, -0.9238795325112867, -0.9238795325112867, -0.9238795325112867 };

__attribute__((target("avx2,fma")))
void radix16_z_w16tgL_fwd_avx2(
    const double * __restrict__ zin,
    const double * __restrict__ zin_unused,
    double       * __restrict__ zout,
    double       * __restrict__ zout_unused,
    const double * tw_re, const double * tw_im,
    size_t Ls, size_t Gs, size_t OLs, size_t OGs, size_t count)
{
    (void)zin_unused; (void)zout_unused; (void)tw_im; (void)Gs; (void)OGs;
    for (size_t k = 0; k + 2 <= count; k += 2) {
        const double *twp = tw_re + (k / 2) * (size_t)120;
        const __m256d v0 = _mm256_loadu_pd(&zin[2*((size_t)8*Ls + k)]);
        const __m256d v1 = _mm256_permute_pd(v0, 0x5);
        const __m256d v2 = _mm256_loadu_pd(&zin[2*((size_t)0*Ls + k)]);
        const __m256d v3 = _mm256_add_pd(v2, v0);
        const __m256d v4 = _mm256_loadu_pd(&zin[2*((size_t)4*Ls + k)]);
        const __m256d v5 = _mm256_permute_pd(v4, 0x5);
        const __m256d v6 = _mm256_loadu_pd(&zin[2*((size_t)12*Ls + k)]);
        const __m256d v7 = _mm256_permute_pd(v6, 0x5);
        const __m256d v8 = _mm256_sub_pd(v2, v0);
        const __m256d v9 = _mm256_add_pd(v4, v6);
        const __m256d v10 = _mm256_loadu_pd(&zin[2*((size_t)14*Ls + k)]);
        const __m256d v11 = _mm256_permute_pd(v10, 0x5);
        const __m256d v12 = _mm256_loadu_pd(&zin[2*((size_t)10*Ls + k)]);
        const __m256d v13 = _mm256_permute_pd(v12, 0x5);
        const __m256d v14 = _mm256_loadu_pd(&zin[2*((size_t)6*Ls + k)]);
        const __m256d v15 = _mm256_permute_pd(v14, 0x5);
        const __m256d v16 = _mm256_loadu_pd(&zin[2*((size_t)2*Ls + k)]);
        const __m256d v17 = _mm256_permute_pd(v16, 0x5);
        const __m256d v18 = _mm256_sub_pd(v4, v6);
        const __m256d v19 = _mm256_add_pd(v14, v10);
        const __m256d v20 = _mm256_sub_pd(v10, v14);
        const __m256d v21 = _mm256_sub_pd(v16, v12);
        const __m256d v22 = _mm256_add_pd(v16, v12);
        const __m256d v23 = _mm256_add_pd(v20, v21);
        const __m256d v24 = _mm256_loadu_pd(&zin[2*((size_t)1*Ls + k)]);
        const __m256d v25 = _mm256_permute_pd(v24, 0x5);
        const __m256d v26 = _mm256_loadu_pd(&zin[2*((size_t)13*Ls + k)]);
        const __m256d v27 = _mm256_permute_pd(v26, 0x5);
        const __m256d v28 = _mm256_loadu_pd(&zin[2*((size_t)9*Ls + k)]);
        const __m256d v29 = _mm256_permute_pd(v28, 0x5);
        const __m256d v30 = _mm256_loadu_pd(&zin[2*((size_t)5*Ls + k)]);
        const __m256d v31 = _mm256_permute_pd(v30, 0x5);
        const __m256d v32 = _mm256_sub_pd(v20, v21);
        const __m256d v33 = _mm256_add_pd(v24, v28);
        const __m256d v34 = _mm256_sub_pd(v24, v28);
        const __m256d v35 = _mm256_sub_pd(v30, v26);
        const __m256d v36 = _mm256_loadu_pd(&zin[2*((size_t)15*Ls + k)]);
        const __m256d v37 = _mm256_permute_pd(v36, 0x5);
        const __m256d v38 = _mm256_loadu_pd(&zin[2*((size_t)11*Ls + k)]);
        const __m256d v39 = _mm256_permute_pd(v38, 0x5);
        const __m256d v40 = _mm256_add_pd(v26, v30);
        const __m256d v41 = _mm256_loadu_pd(&zin[2*((size_t)7*Ls + k)]);
        const __m256d v42 = _mm256_permute_pd(v41, 0x5);
        const __m256d v43 = _mm256_sub_pd(v33, v40);
        const __m256d v44 = _mm256_add_pd(v33, v40);
        const __m256d v45 = _mm256_fnmadd_pd(v34, _TAN8, v35);
        const __m256d v46 = _mm256_loadu_pd(&zin[2*((size_t)3*Ls + k)]);
        const __m256d v47 = _mm256_permute_pd(v46, 0x5);
        const __m256d v48 = _mm256_fmadd_pd(_TAN8, v35, v34);
        const __m256d v49 = _mm256_sub_pd(v36, v41);
        const __m256d v50 = _mm256_add_pd(v36, v41);
        const __m256d v51 = _mm256_add_pd(v38, v46);
        const __m256d v52 = _mm256_sub_pd(v38, v46);
        const __m256d v53 = _mm256_add_pd(v51, v50);
        const __m256d v54 = _mm256_sub_pd(v50, v51);
        const __m256d v55 = _mm256_fmadd_pd(v52, _TAN8, v49);
        const __m256d v56 = _mm256_fnmadd_pd(_TAN8, v49, v52);
        const __m256d v57 = _mm256_sub_pd(v54, v43);
        const __m256d v58 = _mm256_add_pd(v43, v54);
        const __m256d v59 = _mm256_sub_pd(v3, v9);
        const __m256d v60 = _mm256_fmadd_pd(v58, _R2, v59);
        const __m256d v61 = _mm256_fnmadd_pd(_R2, v58, v59);
        const __m256d v62 = _mm256_sub_pd(v19, v22);
        const __m256d v63 = _mm256_fmadd_pd(v57, _R2, v62);
        const __m256d v64 = _mm256_fnmadd_pd(_R2, v57, v62);
        const __m256d v65 = _mm256_permute_pd(v63, 0x5);
        const __m256d v66 = _mm256_xor_pd(v65, _MSK);
        const __m256d v67 = _mm256_sub_pd(v60, v66);
        _mm_storeu_pd(&zout[2*((size_t)k*OLs + 6)], _mm256_castpd256_pd128(v67));
        _mm_storeu_pd(&zout[2*((size_t)(k+1)*OLs + 6)], _mm256_extractf128_pd(v67, 1));
        const __m256d v68 = _mm256_permute_pd(v64, 0x5);
        const __m256d v69 = _mm256_addsub_pd(v61, v68);
        _mm_storeu_pd(&zout[2*((size_t)k*OLs + 2)], _mm256_castpd256_pd128(v69));
        _mm_storeu_pd(&zout[2*((size_t)(k+1)*OLs + 2)], _mm256_extractf128_pd(v69, 1));
        const __m256d v70 = _mm256_add_pd(v9, v3);
        const __m256d v71 = _mm256_addsub_pd(v60, v65);
        _mm_storeu_pd(&zout[2*((size_t)k*OLs + 10)], _mm256_castpd256_pd128(v71));
        _mm_storeu_pd(&zout[2*((size_t)(k+1)*OLs + 10)], _mm256_extractf128_pd(v71, 1));
        const __m256d v72 = _mm256_xor_pd(v68, _MSK);
        const __m256d v73 = _mm256_sub_pd(v61, v72);
        _mm_storeu_pd(&zout[2*((size_t)k*OLs + 14)], _mm256_castpd256_pd128(v73));
        _mm_storeu_pd(&zout[2*((size_t)(k+1)*OLs + 14)], _mm256_extractf128_pd(v73, 1));
        const __m256d v74 = _mm256_add_pd(v22, v19);
        const __m256d v75 = _mm256_fmadd_pd(v23, _R2, v8);
        const __m256d v76 = _mm256_fnmadd_pd(_R2, v23, v8);
        const __m256d v77 = _mm256_add_pd(v56, v45);
        const __m256d v78 = _mm256_fmadd_pd(v77, _CP8, v75);
        const __m256d v79 = _mm256_fnmadd_pd(_CP8, v77, v75);
        const __m256d v80 = _mm256_fnmadd_pd(v32, _R2, v18);
        const __m256d v81 = _mm256_sub_pd(v55, v48);
        const __m256d v82 = _mm256_fmadd_pd(v81, _CP8, v80);
        const __m256d v83 = _mm256_fnmadd_pd(_CP8, v81, v80);
        const __m256d v84 = _mm256_permute_pd(v82, 0x5);
        const __m256d v85 = _mm256_xor_pd(v84, _MSK);
        const __m256d v86 = _mm256_sub_pd(v78, v85);
        _mm_storeu_pd(&zout[2*((size_t)k*OLs + 5)], _mm256_castpd256_pd128(v86));
        _mm_storeu_pd(&zout[2*((size_t)(k+1)*OLs + 5)], _mm256_extractf128_pd(v86, 1));
        const __m256d v87 = _mm256_permute_pd(v83, 0x5);
        const __m256d v88 = _mm256_xor_pd(v87, _MSK);
        const __m256d v89 = _mm256_sub_pd(v79, v88);
        _mm_storeu_pd(&zout[2*((size_t)k*OLs + 13)], _mm256_castpd256_pd128(v89));
        _mm_storeu_pd(&zout[2*((size_t)(k+1)*OLs + 13)], _mm256_extractf128_pd(v89, 1));
        const __m256d v90 = _mm256_sub_pd(v45, v56);
        const __m256d v91 = _mm256_addsub_pd(v78, v84);
        _mm_storeu_pd(&zout[2*((size_t)k*OLs + 11)], _mm256_castpd256_pd128(v91));
        _mm_storeu_pd(&zout[2*((size_t)(k+1)*OLs + 11)], _mm256_extractf128_pd(v91, 1));
        const __m256d v92 = _mm256_fmadd_pd(_R2, v32, v18);
        const __m256d v93 = _mm256_addsub_pd(v79, v87);
        _mm_storeu_pd(&zout[2*((size_t)k*OLs + 3)], _mm256_castpd256_pd128(v93));
        _mm_storeu_pd(&zout[2*((size_t)(k+1)*OLs + 3)], _mm256_extractf128_pd(v93, 1));
        const __m256d v94 = _mm256_add_pd(v55, v48);
        const __m256d v95 = _mm256_add_pd(v70, v74);
        const __m256d v96 = _mm256_sub_pd(v70, v74);
        const __m256d v97 = _mm256_add_pd(v44, v53);
        const __m256d v98 = _mm256_sub_pd(v53, v44);
        const __m256d v99 = _mm256_sub_pd(v95, v97);
        _mm_storeu_pd(&zout[2*((size_t)k*OLs + 8)], _mm256_castpd256_pd128(v99));
        _mm_storeu_pd(&zout[2*((size_t)(k+1)*OLs + 8)], _mm256_extractf128_pd(v99, 1));
        const __m256d v100 = _mm256_permute_pd(v98, 0x5);
        const __m256d v101 = _mm256_addsub_pd(v96, v100);
        _mm_storeu_pd(&zout[2*((size_t)k*OLs + 4)], _mm256_castpd256_pd128(v101));
        _mm_storeu_pd(&zout[2*((size_t)(k+1)*OLs + 4)], _mm256_extractf128_pd(v101, 1));
        const __m256d v102 = _mm256_add_pd(v95, v97);
        _mm_storeu_pd(&zout[2*((size_t)k*OLs + 0)], _mm256_castpd256_pd128(v102));
        _mm_storeu_pd(&zout[2*((size_t)(k+1)*OLs + 0)], _mm256_extractf128_pd(v102, 1));
        const __m256d v103 = _mm256_xor_pd(v100, _MSK);
        const __m256d v104 = _mm256_sub_pd(v96, v103);
        _mm_storeu_pd(&zout[2*((size_t)k*OLs + 12)], _mm256_castpd256_pd128(v104));
        _mm_storeu_pd(&zout[2*((size_t)(k+1)*OLs + 12)], _mm256_extractf128_pd(v104, 1));
        const __m256d v105 = _mm256_fmadd_pd(v94, _CP8, v76);
        const __m256d v106 = _mm256_fnmadd_pd(_CP8, v94, v76);
        const __m256d v107 = _mm256_fmadd_pd(v90, _CP8, v92);
        const __m256d v108 = _mm256_fnmadd_pd(_CP8, v90, v92);
        const __m256d v109 = _mm256_permute_pd(v107, 0x5);
        const __m256d v110 = _mm256_xor_pd(v109, _MSK);
        const __m256d v111 = _mm256_sub_pd(v105, v110);
        _mm_storeu_pd(&zout[2*((size_t)k*OLs + 9)], _mm256_castpd256_pd128(v111));
        _mm_storeu_pd(&zout[2*((size_t)(k+1)*OLs + 9)], _mm256_extractf128_pd(v111, 1));
        const __m256d v112 = _mm256_permute_pd(v108, 0x5);
        const __m256d v113 = _mm256_addsub_pd(v106, v112);
        _mm_storeu_pd(&zout[2*((size_t)k*OLs + 15)], _mm256_castpd256_pd128(v113));
        _mm_storeu_pd(&zout[2*((size_t)(k+1)*OLs + 15)], _mm256_extractf128_pd(v113, 1));
        const __m256d v114 = _mm256_addsub_pd(v105, v109);
        _mm_storeu_pd(&zout[2*((size_t)k*OLs + 7)], _mm256_castpd256_pd128(v114));
        _mm_storeu_pd(&zout[2*((size_t)(k+1)*OLs + 7)], _mm256_extractf128_pd(v114, 1));
        const __m256d v115 = _mm256_xor_pd(v112, _MSK);
        const __m256d v116 = _mm256_sub_pd(v106, v115);
        _mm_storeu_pd(&zout[2*((size_t)k*OLs + 1)], _mm256_castpd256_pd128(v116));
        _mm_storeu_pd(&zout[2*((size_t)(k+1)*OLs + 1)], _mm256_extractf128_pd(v116, 1));
    }
}
