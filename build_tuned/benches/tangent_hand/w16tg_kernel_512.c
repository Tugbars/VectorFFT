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

__attribute__((target("avx512f,avx512vl,fma")))
void radix16_z_w16tg_fwd_avx2(
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
        const __m256d v2 = _mm256_mul_pd(v1, _mm256_loadu_pd(&twp[60]));
        const __m256d v3 = _mm256_fmsub_pd(v0, _mm256_loadu_pd(&twp[56]), v2);
        const __m256d v4 = _mm256_loadu_pd(&zin[2*((size_t)0*Ls + k)]);
        const __m256d v5 = _mm256_add_pd(v4, v3);
        const __m256d v6 = _mm256_loadu_pd(&zin[2*((size_t)4*Ls + k)]);
        const __m256d v7 = _mm256_permute_pd(v6, 0x5);
        const __m256d v8 = _mm256_mul_pd(v7, _mm256_loadu_pd(&twp[28]));
        const __m256d v9 = _mm256_fmsub_pd(v6, _mm256_loadu_pd(&twp[24]), v8);
        const __m256d v10 = _mm256_loadu_pd(&zin[2*((size_t)12*Ls + k)]);
        const __m256d v11 = _mm256_permute_pd(v10, 0x5);
        const __m256d v12 = _mm256_mul_pd(v11, _mm256_loadu_pd(&twp[92]));
        const __m256d v13 = _mm256_sub_pd(v4, v3);
        const __m256d v14 = _mm256_fmsub_pd(v10, _mm256_loadu_pd(&twp[88]), v12);
        const __m256d v15 = _mm256_add_pd(v9, v14);
        const __m256d v16 = _mm256_loadu_pd(&zin[2*((size_t)14*Ls + k)]);
        const __m256d v17 = _mm256_permute_pd(v16, 0x5);
        const __m256d v18 = _mm256_mul_pd(v17, _mm256_loadu_pd(&twp[108]));
        const __m256d v19 = _mm256_fmsub_pd(v16, _mm256_loadu_pd(&twp[104]), v18);
        const __m256d v20 = _mm256_loadu_pd(&zin[2*((size_t)10*Ls + k)]);
        const __m256d v21 = _mm256_permute_pd(v20, 0x5);
        const __m256d v22 = _mm256_mul_pd(v21, _mm256_loadu_pd(&twp[76]));
        const __m256d v23 = _mm256_fmsub_pd(v20, _mm256_loadu_pd(&twp[72]), v22);
        const __m256d v24 = _mm256_loadu_pd(&zin[2*((size_t)6*Ls + k)]);
        const __m256d v25 = _mm256_permute_pd(v24, 0x5);
        const __m256d v26 = _mm256_mul_pd(v25, _mm256_loadu_pd(&twp[44]));
        const __m256d v27 = _mm256_fmsub_pd(v24, _mm256_loadu_pd(&twp[40]), v26);
        const __m256d v28 = _mm256_loadu_pd(&zin[2*((size_t)2*Ls + k)]);
        const __m256d v29 = _mm256_permute_pd(v28, 0x5);
        const __m256d v30 = _mm256_mul_pd(v29, _mm256_loadu_pd(&twp[12]));
        const __m256d v31 = _mm256_fmsub_pd(v28, _mm256_loadu_pd(&twp[8]), v30);
        const __m256d v32 = _mm256_sub_pd(v9, v14);
        const __m256d v33 = _mm256_add_pd(v27, v19);
        const __m256d v34 = _mm256_sub_pd(v19, v27);
        const __m256d v35 = _mm256_sub_pd(v31, v23);
        const __m256d v36 = _mm256_add_pd(v31, v23);
        const __m256d v37 = _mm256_add_pd(v34, v35);
        const __m256d v38 = _mm256_loadu_pd(&zin[2*((size_t)1*Ls + k)]);
        const __m256d v39 = _mm256_permute_pd(v38, 0x5);
        const __m256d v40 = _mm256_mul_pd(v39, _mm256_loadu_pd(&twp[4]));
        const __m256d v41 = _mm256_fmsub_pd(v38, _mm256_loadu_pd(&twp[0]), v40);
        const __m256d v42 = _mm256_loadu_pd(&zin[2*((size_t)13*Ls + k)]);
        const __m256d v43 = _mm256_permute_pd(v42, 0x5);
        const __m256d v44 = _mm256_mul_pd(v43, _mm256_loadu_pd(&twp[100]));
        const __m256d v45 = _mm256_fmsub_pd(v42, _mm256_loadu_pd(&twp[96]), v44);
        const __m256d v46 = _mm256_loadu_pd(&zin[2*((size_t)9*Ls + k)]);
        const __m256d v47 = _mm256_permute_pd(v46, 0x5);
        const __m256d v48 = _mm256_mul_pd(v47, _mm256_loadu_pd(&twp[68]));
        const __m256d v49 = _mm256_fmsub_pd(v46, _mm256_loadu_pd(&twp[64]), v48);
        const __m256d v50 = _mm256_loadu_pd(&zin[2*((size_t)5*Ls + k)]);
        const __m256d v51 = _mm256_permute_pd(v50, 0x5);
        const __m256d v52 = _mm256_mul_pd(v51, _mm256_loadu_pd(&twp[36]));
        const __m256d v53 = _mm256_sub_pd(v34, v35);
        const __m256d v54 = _mm256_fmsub_pd(v50, _mm256_loadu_pd(&twp[32]), v52);
        const __m256d v55 = _mm256_add_pd(v41, v49);
        const __m256d v56 = _mm256_sub_pd(v41, v49);
        const __m256d v57 = _mm256_sub_pd(v54, v45);
        const __m256d v58 = _mm256_loadu_pd(&zin[2*((size_t)15*Ls + k)]);
        const __m256d v59 = _mm256_permute_pd(v58, 0x5);
        const __m256d v60 = _mm256_mul_pd(v59, _mm256_loadu_pd(&twp[116]));
        const __m256d v61 = _mm256_fmsub_pd(v58, _mm256_loadu_pd(&twp[112]), v60);
        const __m256d v62 = _mm256_loadu_pd(&zin[2*((size_t)11*Ls + k)]);
        const __m256d v63 = _mm256_permute_pd(v62, 0x5);
        const __m256d v64 = _mm256_mul_pd(v63, _mm256_loadu_pd(&twp[84]));
        const __m256d v65 = _mm256_add_pd(v45, v54);
        const __m256d v66 = _mm256_fmsub_pd(v62, _mm256_loadu_pd(&twp[80]), v64);
        const __m256d v67 = _mm256_loadu_pd(&zin[2*((size_t)7*Ls + k)]);
        const __m256d v68 = _mm256_permute_pd(v67, 0x5);
        const __m256d v69 = _mm256_mul_pd(v68, _mm256_loadu_pd(&twp[52]));
        const __m256d v70 = _mm256_sub_pd(v55, v65);
        const __m256d v71 = _mm256_add_pd(v55, v65);
        const __m256d v72 = _mm256_fmsub_pd(v67, _mm256_loadu_pd(&twp[48]), v69);
        const __m256d v73 = _mm256_fnmadd_pd(v56, _TAN8, v57);
        const __m256d v74 = _mm256_loadu_pd(&zin[2*((size_t)3*Ls + k)]);
        const __m256d v75 = _mm256_permute_pd(v74, 0x5);
        const __m256d v76 = _mm256_mul_pd(v75, _mm256_loadu_pd(&twp[20]));
        const __m256d v77 = _mm256_fmadd_pd(_TAN8, v57, v56);
        const __m256d v78 = _mm256_fmsub_pd(v74, _mm256_loadu_pd(&twp[16]), v76);
        const __m256d v79 = _mm256_sub_pd(v61, v72);
        const __m256d v80 = _mm256_add_pd(v61, v72);
        const __m256d v81 = _mm256_add_pd(v66, v78);
        const __m256d v82 = _mm256_sub_pd(v66, v78);
        const __m256d v83 = _mm256_add_pd(v81, v80);
        const __m256d v84 = _mm256_sub_pd(v80, v81);
        const __m256d v85 = _mm256_fmadd_pd(v82, _TAN8, v79);
        const __m256d v86 = _mm256_fnmadd_pd(_TAN8, v79, v82);
        const __m256d v87 = _mm256_sub_pd(v84, v70);
        const __m256d v88 = _mm256_add_pd(v70, v84);
        const __m256d v89 = _mm256_sub_pd(v5, v15);
        const __m256d v90 = _mm256_fmadd_pd(v88, _R2, v89);
        const __m256d v91 = _mm256_fnmadd_pd(_R2, v88, v89);
        const __m256d v92 = _mm256_sub_pd(v33, v36);
        const __m256d v93 = _mm256_fmadd_pd(v87, _R2, v92);
        const __m256d v94 = _mm256_fnmadd_pd(_R2, v87, v92);
        const __m256d v95 = _mm256_permute_pd(v93, 0x5);
        const __m256d v96 = _mm256_xor_pd(v95, _MSK);
        const __m256d v97 = _mm256_sub_pd(v90, v96);
        _mm256_storeu_pd(&zout[2*((size_t)6*OLs + k)], v97);
        const __m256d v98 = _mm256_permute_pd(v94, 0x5);
        const __m256d v99 = _mm256_addsub_pd(v91, v98);
        _mm256_storeu_pd(&zout[2*((size_t)2*OLs + k)], v99);
        const __m256d v100 = _mm256_add_pd(v15, v5);
        const __m256d v101 = _mm256_addsub_pd(v90, v95);
        _mm256_storeu_pd(&zout[2*((size_t)10*OLs + k)], v101);
        const __m256d v102 = _mm256_xor_pd(v98, _MSK);
        const __m256d v103 = _mm256_sub_pd(v91, v102);
        _mm256_storeu_pd(&zout[2*((size_t)14*OLs + k)], v103);
        const __m256d v104 = _mm256_add_pd(v36, v33);
        const __m256d v105 = _mm256_fmadd_pd(v37, _R2, v13);
        const __m256d v106 = _mm256_fnmadd_pd(_R2, v37, v13);
        const __m256d v107 = _mm256_add_pd(v86, v73);
        const __m256d v108 = _mm256_fmadd_pd(v107, _CP8, v105);
        const __m256d v109 = _mm256_fnmadd_pd(_CP8, v107, v105);
        const __m256d v110 = _mm256_fnmadd_pd(v53, _R2, v32);
        const __m256d v111 = _mm256_sub_pd(v85, v77);
        const __m256d v112 = _mm256_fmadd_pd(v111, _CP8, v110);
        const __m256d v113 = _mm256_fnmadd_pd(_CP8, v111, v110);
        const __m256d v114 = _mm256_permute_pd(v112, 0x5);
        const __m256d v115 = _mm256_xor_pd(v114, _MSK);
        const __m256d v116 = _mm256_sub_pd(v108, v115);
        _mm256_storeu_pd(&zout[2*((size_t)5*OLs + k)], v116);
        const __m256d v117 = _mm256_permute_pd(v113, 0x5);
        const __m256d v118 = _mm256_xor_pd(v117, _MSK);
        const __m256d v119 = _mm256_sub_pd(v109, v118);
        _mm256_storeu_pd(&zout[2*((size_t)13*OLs + k)], v119);
        const __m256d v120 = _mm256_sub_pd(v73, v86);
        const __m256d v121 = _mm256_addsub_pd(v108, v114);
        _mm256_storeu_pd(&zout[2*((size_t)11*OLs + k)], v121);
        const __m256d v122 = _mm256_fmadd_pd(_R2, v53, v32);
        const __m256d v123 = _mm256_addsub_pd(v109, v117);
        _mm256_storeu_pd(&zout[2*((size_t)3*OLs + k)], v123);
        const __m256d v124 = _mm256_add_pd(v85, v77);
        const __m256d v125 = _mm256_add_pd(v100, v104);
        const __m256d v126 = _mm256_sub_pd(v100, v104);
        const __m256d v127 = _mm256_add_pd(v71, v83);
        const __m256d v128 = _mm256_sub_pd(v83, v71);
        const __m256d v129 = _mm256_sub_pd(v125, v127);
        _mm256_storeu_pd(&zout[2*((size_t)8*OLs + k)], v129);
        const __m256d v130 = _mm256_permute_pd(v128, 0x5);
        const __m256d v131 = _mm256_addsub_pd(v126, v130);
        _mm256_storeu_pd(&zout[2*((size_t)4*OLs + k)], v131);
        const __m256d v132 = _mm256_add_pd(v125, v127);
        _mm256_storeu_pd(&zout[2*((size_t)0*OLs + k)], v132);
        const __m256d v133 = _mm256_xor_pd(v130, _MSK);
        const __m256d v134 = _mm256_sub_pd(v126, v133);
        _mm256_storeu_pd(&zout[2*((size_t)12*OLs + k)], v134);
        const __m256d v135 = _mm256_fmadd_pd(v124, _CP8, v106);
        const __m256d v136 = _mm256_fnmadd_pd(_CP8, v124, v106);
        const __m256d v137 = _mm256_fmadd_pd(v120, _CP8, v122);
        const __m256d v138 = _mm256_fnmadd_pd(_CP8, v120, v122);
        const __m256d v139 = _mm256_permute_pd(v137, 0x5);
        const __m256d v140 = _mm256_xor_pd(v139, _MSK);
        const __m256d v141 = _mm256_sub_pd(v135, v140);
        _mm256_storeu_pd(&zout[2*((size_t)9*OLs + k)], v141);
        const __m256d v142 = _mm256_permute_pd(v138, 0x5);
        const __m256d v143 = _mm256_addsub_pd(v136, v142);
        _mm256_storeu_pd(&zout[2*((size_t)15*OLs + k)], v143);
        const __m256d v144 = _mm256_addsub_pd(v135, v139);
        _mm256_storeu_pd(&zout[2*((size_t)7*OLs + k)], v144);
        const __m256d v145 = _mm256_xor_pd(v142, _MSK);
        const __m256d v146 = _mm256_sub_pd(v136, v145);
        _mm256_storeu_pd(&zout[2*((size_t)1*OLs + k)], v146);
    }
}
