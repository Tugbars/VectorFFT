/* il2d_real_axis.h -- THE REAL AXIS ON N1 (2026-10-07): a whole-plan FORM of
 * the interleaved 2D r2c plan at an even N1 and an odd N2
 * (docs/roadmap/il2d_real_axis_n1.md).
 *
 * The standard walk puts the real transform on the rows (N1 real rows of N2
 * points: the row plan, at an odd N2 the odd door's engine per row) and the
 * complex chain on the columns. This form puts the real axis on N1:
 *   1. PACK    W[m][n] = x[2m][n] + i x[2m+1][n], m < M = N1/2: a complex
 *              M x N2 plane at pitch N2.
 *   2. COLUMNS the c2c column chain of length M down every column of W, in
 *              place, the SCRAMBLED class (the chain the 2D c2c tier would
 *              serve at (M, N2), raced in this plan's role on the form's own
 *              store, rax_* on the real row; a prime M takes the column
 *              Bluestein or the turned prime pass, natural out).
 *   3. FOLD    for every column, with Z the column spectrum of W (read through
 *              the chain's permutation, which un-scrambles it for free):
 *              A[0] = Re Z[0] + Im Z[0], A[M] = Re Z[0] - Im Z[0], and for
 *              k = 1..M/2: E = (Z[k] + conj Z[M-k]) / 2, O = -i (Z[k] - conj
 *              Z[M-k]) / 2, A[k] = E + w^k O, A[M-k] = conj(E - w^k O),
 *              w = e^{-2 pi i / N1}. A = the real column spectra, rows 0..M,
 *              an (M+1) x N2 plane in natural order.
 *   4. ROWS    c2c(N2) on the M+1 rows of A: a transform-contiguous batch at
 *              N2 x (M+1), raced in this plan's role (raxr_* on the real row),
 *              out of place into T.
 *   5. OUT     CCE row k (k <= M) = bins 0..hp1-1 of T[k]; CCE row N1-k
 *              (0 < k < M) = conj of bins (N2 - f) mod N2 of T[k].
 * The row work halves (M+1 complex transforms of N2 instead of N1 real ones,
 * which at a prime N2 each cost nearly a c2c); the fold is elementwise across
 * a row. FFTW has no such form (rank>=2 rdft2 takes the real transform on the
 * last dimension).
 *
 * PLANNING. A candidate, never a default: raced at create against the
 * standard walk on the whole transform (the plan's own row and column plans,
 * as it serves them), its plane gated against the standard walk's at 1e-10,
 * the race's margin (VFFT_RACE_HYST) toward the standard walk; banked
 * raxis=n1 (the form serves) or raxis=n2 (the standard walk) on the real row
 * at the plan's T. VFFT_IL2D_RAXIS=n1|n2 pins (beats wisdom, never banks).
 * Admission is the policy's law (vfft_policy_il2d_raxis_ok,
 * planning/policy_il.h: r2c, interleaved, out of place, one thread, an even
 * N1 >= 4, an odd N2) with the standard walk present to race against. The
 * threaded form and the c2r twin are their own pieces (the owner's rule: r2c
 * and c2r are separate).
 *
 * Included after il2d_real_plan.h (the row and column plans the standard
 * walk's arm runs) and before vfft_execute.h (the dispatch and the destroy).
 */
#ifndef VFFT_IL2D_REAL_AXIS_H
#define VFFT_IL2D_REAL_AXIS_H

/* the form's admission at this plan: the policy's law of the request
 * (planning/policy_il.h), and the standard walk present to race against */
static int _il2d_rax_admits(const struct vfft_plan_s *h, const vfft_config_t *cfg)
{
    return h->il2d_row && h->il2d_oddn2 && vfft_policy_il2d_raxis_ok(cfg, h->nthreads, h->N, h->N2);
}

/* everything the form owns, back to nothing (the plan's standard walk untouched) */
static void _il2d_rax_free(struct vfft_plan_s *h)
{
    vfft_aligned_free(h->il2d_rax_W);
    vfft_aligned_free(h->il2d_rax_A);
    vfft_aligned_free(h->il2d_rax_T);
    vfft_aligned_free(h->il2d_rax_tw);
    free(h->il2d_rax_perm);
    if (h->il2d_rax_row)
        vfft_destroy((vfft_plan)h->il2d_rax_row);
    vfft_child_store_free(h->il2d_rax_col.tpcS);
    h->il2d_rax_col.tpcS = NULL;
    _il2d_col_free(&h->il2d_rax_col);   /* the chain's tables, the Bluestein's planes; zeroes the descriptor */
    vfft_child_store_free(h->il2d_raxS);
    vfft_child_store_free(h->il2d_rax_rowS);
    h->il2d_rax_W = h->il2d_rax_A = h->il2d_rax_T = h->il2d_rax_tw = NULL;
    h->il2d_rax_perm = NULL;
    h->il2d_rax_row = NULL;
    h->il2d_raxS = h->il2d_rax_rowS = NULL;
    h->il2d_rax_on = 0;
    h->il2d_rax_M = 0;
}

/* ── the passes ─────────────────────────────────────────────────────── */

/* 1. PACK: row pairs into one complex row */
static void _il2d_rax_pack(const double *x, double *W, size_t M, size_t N2)
{
    size_t m, n;
    for (m = 0; m < M; m++)
    {
        const double *a = x + 2 * m * N2, *b = a + N2;
        double *w = W + 2 * m * N2;
        n = 0;
#if defined(__AVX2__)
        for (; n + 4 <= N2; n += 4)
        {
            const __m256d va = _mm256_loadu_pd(a + n), vb = _mm256_loadu_pd(b + n);
            const __m256d lo = _mm256_unpacklo_pd(va, vb), hi = _mm256_unpackhi_pd(va, vb);
            _mm256_storeu_pd(w + 2 * n, _mm256_permute2f128_pd(lo, hi, 0x20));
            _mm256_storeu_pd(w + 2 * n + 4, _mm256_permute2f128_pd(lo, hi, 0x31));
        }
#endif
        for (; n < N2; n++)
        {
            w[2 * n] = a[n];
            w[2 * n + 1] = b[n];
        }
    }
}

/* 3. FOLD: W's column spectra (rows read through perm) -> A, rows 0..M natural */
static void _il2d_rax_fold(const double *W, double *A, size_t M, size_t N2, const int *perm, const double *tw)
{
    const double *Z0 = W + 2 * (perm ? (size_t)perm[0] : 0) * N2;
    double *A0 = A, *AM = A + 2 * M * N2;
    size_t k, n;
    for (n = 0; n < N2; n++)
    {
        const double r = Z0[2 * n], i = Z0[2 * n + 1];
        A0[2 * n] = r + i;
        A0[2 * n + 1] = 0.0;
        AM[2 * n] = r - i;
        AM[2 * n + 1] = 0.0;
    }
    for (k = 1; 2 * k <= M; k++)
    {
        const size_t rk = perm ? (size_t)perm[k] : k, rm = perm ? (size_t)perm[M - k] : M - k;
        const double *Pk = W + 2 * rk * N2, *Pm = W + 2 * rm * N2;
        double *Ak = A + 2 * k * N2, *Am = A + 2 * (M - k) * N2;
        const double wr = tw[2 * k], wi = tw[2 * k + 1];
        const int mirror = (2 * k < M);
        n = 0;
#if defined(__AVX2__)
        {
            const __m256d cj = _mm256_setr_pd(0.0, -0.0, 0.0, -0.0), half = _mm256_set1_pd(0.5);
            const __m256d vwr = _mm256_set1_pd(wr), vwi = _mm256_setr_pd(-wi, wi, -wi, wi);
            for (; n + 2 <= N2; n += 2)
            {
                const __m256d P = _mm256_loadu_pd(Pk + 2 * n), Q = _mm256_xor_pd(_mm256_loadu_pd(Pm + 2 * n), cj);
                const __m256d E = _mm256_mul_pd(half, _mm256_add_pd(P, Q)), D = _mm256_mul_pd(half, _mm256_sub_pd(P, Q));
                const __m256d O = _mm256_xor_pd(_mm256_permute_pd(D, 0x5), cj);                          /* -i D */
                const __m256d T = _mm256_fmadd_pd(O, vwr, _mm256_mul_pd(_mm256_permute_pd(O, 0x5), vwi)); /* w^k O */
                _mm256_storeu_pd(Ak + 2 * n, _mm256_add_pd(E, T));
                if (mirror)
                    _mm256_storeu_pd(Am + 2 * n, _mm256_xor_pd(_mm256_sub_pd(E, T), cj));
            }
        }
#endif
        for (; n < N2; n++)
        {
            const double pr = Pk[2 * n], pi = Pk[2 * n + 1], qr = Pm[2 * n], qi = -Pm[2 * n + 1];
            const double er = 0.5 * (pr + qr), ei = 0.5 * (pi + qi), dr = 0.5 * (pr - qr), di = 0.5 * (pi - qi);
            const double orr = di, oi = -dr;   /* -i D */
            const double tr = orr * wr - oi * wi, ti = orr * wi + oi * wr;
            Ak[2 * n] = er + tr;
            Ak[2 * n + 1] = ei + ti;
            if (mirror)
            {
                Am[2 * n] = er - tr;
                Am[2 * n + 1] = -(ei - ti);
            }
        }
    }
}

/* 5. OUT: dst[j] = conj(src[(n - j) mod n]) for j < cnt (the mirror row) */
static void _il2d_rax_mirror(double *dst, const double *src, size_t n, size_t cnt)
{
    size_t j = 0;
    if (cnt)
    {
        dst[0] = src[0];
        dst[1] = -src[1];
        j = 1;
    }
#if defined(__AVX2__)
    {
        const __m256d cj = _mm256_setr_pd(0.0, -0.0, 0.0, -0.0);
        for (; j + 2 <= cnt; j += 2)
        {   /* dst[j], dst[j+1] <- conj src[n-j], conj src[n-j-1] */
            const __m256d v = _mm256_loadu_pd(src + 2 * (n - j - 1));
            _mm256_storeu_pd(dst + 2 * j, _mm256_xor_pd(_mm256_permute2f128_pd(v, v, 0x01), cj));
        }
    }
#endif
    for (; j < cnt; j++)
    {
        dst[2 * j] = src[2 * (n - j)];
        dst[2 * j + 1] = -src[2 * (n - j) + 1];
    }
}
static void _il2d_rax_out(const double *T, double *z, size_t N1, size_t N2, size_t M, size_t hp1)
{
    size_t k;
    for (k = 0; k <= M; k++)
        memcpy(z + 2 * k * hp1, T + 2 * k * N2, 2 * hp1 * sizeof(double));
    for (k = 1; k < M; k++)
        _il2d_rax_mirror(z + 2 * (N1 - k) * hp1, T + 2 * k * N2, N2, hp1);
}

/* THE FORM'S EXECUTE: x (N1 x N2 real) -> z (the N1 x hp1 CCE plane) */
static void _il2d_rax_exec_fwd(struct vfft_plan_s *h, const double *x, double *z)
{
    const size_t N1 = (size_t)h->N, N2 = (size_t)h->N2, M = (size_t)h->il2d_rax_M, hp1 = N2 / 2 + 1;
    _il2d_rax_pack(x, h->il2d_rax_W, M, N2);
    _il2d_col_exec(&h->il2d_rax_col, h->il2d_rax_W, h->il2d_rax_W, /*reverse=*/0);
    _il2d_rax_fold(h->il2d_rax_W, h->il2d_rax_A, M, N2, h->il2d_rax_perm, h->il2d_rax_tw);
    vfft_execute((vfft_plan)h->il2d_rax_row, VFFT_FORWARD, h->il2d_rax_A, NULL, h->il2d_rax_T, NULL);
    _il2d_rax_out(h->il2d_rax_T, z, N1, N2, M, hp1);
}

/* ── the build: the two children in role, the planes, the twiddles ───── */
static int _il2d_rax_build(struct vfft_plan_s *h, struct vfft_wisdom_s *W, const vfft_config_t *cfg, int ord, int T)
{
    const int N1 = h->N, N2 = h->N2, M = N1 / 2;
    const size_t rn2 = (size_t)N2;
    const char *log = getenv("VFFT_IL2D_LOG");
    struct vfft_wisdom_s *const W0 = _il2d_blu_ctx.W;   /* the create in progress: restored after the children */
    const vfft_config_t *const c0 = _il2d_blu_ctx.cfg;
    vw2_ilcol_key_t pck;
    vw2_key_t pk;
    int k;
    memset(&pck, 0, sizeof pck);
    pck.rank = 2; pck.n0 = N1; pck.n1 = N2; pck.ord = ord; pck.real = 1; pck.nthreads = T;
    vw2__ilcol_key(&pck, &pk);
    if (!h->il2d_raxS)
        h->il2d_raxS = vfft_child_store_for(W ? &W->vw2 : NULL, &pk, "rax_");
    if (!h->il2d_rax_rowS)
        h->il2d_rax_rowS = vfft_child_store_for(W ? &W->vw2 : NULL, &pk, "raxr_");
    if (!h->il2d_raxS || !h->il2d_rax_rowS)
        return 0;
    h->il2d_rax_M = M;
    {   /* 2. the column chain at M down the N2 columns, in role: the shared column
         * builder against the form's private store (rax_*), the scrambled class,
         * one thread; nothing of it is persisted from here (wisdom_write = 0) */
        vfft_config_t cc = *cfg;
        vw2_ilcol_key_t ck;
        char forms[64];
        int bwl = -1, btf = -1, bro = -1, bcmt = -1, bcmtt = -1, bblu = -1, ok;
        cc.wisdom = (vfft_wisdom *)h->il2d_raxS;
        cc.wisdom_write = 0;
        cc.nthreads = 1;
        memset(&ck, 0, sizeof ck);
        ck.rank = 2; ck.n0 = M; ck.n1 = N2; ck.ord = VW2_ORD_SCR; ck.nthreads = 1;
        memset(&h->il2d_rax_col, 0, sizeof h->il2d_rax_col);
        forms[0] = 0;
        _il2d_blu_ctx.W = h->il2d_raxS;
        _il2d_blu_ctx.cfg = &cc;
        ok = _il2d_col_build(h->il2d_raxS, &cc, &ck, M, rn2, /*nat_req=*/0, &h->il2d_rax_col, forms, sizeof forms,
                             &bwl, &btf, &bro, &bcmt, &bcmtt, &bblu);
        _il2d_blu_ctx.W = W0;
        _il2d_blu_ctx.cfg = c0;
        if (!ok)
        {
            if (log)
                fprintf(stderr, "[il2d-real] raxis %dx%d: no column chain at M=%d -- the form does not build\n", N1, N2, M);
            return 0;
        }
        h->il2d_rax_col.N = M;
        h->il2d_rax_col.rn = rn2;
        h->il2d_rax_col.wl = 0;
        h->il2d_rax_col.cut = 0;
        h->il2d_rax_col.nat = 0;
        h->il2d_rax_col.colmt = 0;
    }
    if (!h->il2d_rax_col.blu && h->il2d_rax_col.nst >= 2)
    {   /* the scrambled pass's row order: the chain's digit-reversal comb (natural row of
         * each pass row); the fold reads row k at the pass row that holds it */
        int *np = _il2d_nat_perm(h->il2d_rax_col.R, h->il2d_rax_col.nst, M);
        if (!np)
        {
            if (log)
                fprintf(stderr, "[il2d-real] raxis %dx%d: the chain's permutation at M=%d does not derive -- the form does not build\n", N1, N2, M);
            return 0;
        }
        h->il2d_rax_perm = (int *)malloc((size_t)M * sizeof(int));
        if (!h->il2d_rax_perm)
        {
            free(np);
            return 0;
        }
        for (k = 0; k < M; k++)
            h->il2d_rax_perm[np[k]] = k;
        free(np);
    }
    {   /* 4. the c2c(N2) batch over the M+1 folded rows, in role (raxr_*): transform-
         * contiguous, natural, out of place, one thread */
        vfft_config_t rc;
        memset(&rc, 0, sizeof rc);
        rc.transform = VFFT_C2C;
        rc.placement = VFFT_OUTOFPLACE;
        rc.rigor = cfg->rigor;
        rc.dims = 1;
        rc.n[0] = N2;
        rc.howmany = (size_t)M + 1;
        rc.batch_geom = VFFT_BATCH_TRANSFORM_CONTIGUOUS;
        rc.layout = VFFT_LAYOUT_INTERLEAVED;
        rc.order = VFFT_ORDER_NATURAL;
        rc.nthreads = 1;
        rc.wisdom = (vfft_wisdom *)h->il2d_rax_rowS;
        rc.wisdom_write = 0;
        rc.recalibrate = cfg->recalibrate;
        h->il2d_rax_row = (struct vfft_plan_s *)vfft_create(&rc);
        _il2d_blu_ctx.W = W0;   /* a nested create sets the context to its own store */
        _il2d_blu_ctx.cfg = c0;
        if (!h->il2d_rax_row)
        {
            if (log)
                fprintf(stderr, "[il2d-real] raxis %dx%d: the c2c batch at %d x %d does not build -- the form does not build\n", N1, N2, N2, M + 1);
            return 0;
        }
    }
    h->il2d_rax_W = (double *)vfft_aligned_alloc((2 * (size_t)M * rn2 + 8) * sizeof(double));
    h->il2d_rax_A = (double *)vfft_aligned_alloc((2 * ((size_t)M + 1) * rn2 + 8) * sizeof(double));
    h->il2d_rax_T = (double *)vfft_aligned_alloc((2 * ((size_t)M + 1) * rn2 + 8) * sizeof(double));
    h->il2d_rax_tw = (double *)vfft_aligned_alloc(2 * ((size_t)M + 1) * sizeof(double));
    if (!h->il2d_rax_W || !h->il2d_rax_A || !h->il2d_rax_T || !h->il2d_rax_tw)
        return 0;
    for (k = 0; k <= M; k++)
    {   /* w^k = e^{-2 pi i k / N1}, once-rounded (common/math/tw_exact.h) */
        double c, s;
        vfft_cs2pi_exact((long long)k, (long long)N1, &c, &s);
        h->il2d_rax_tw[2 * k] = c;
        h->il2d_rax_tw[2 * k + 1] = -s;
    }
    return 1;
}

/* the verdict onto the real row (the row is made when the cell has none yet) */
static void _il2d_rax_bank(struct vfft_wisdom_s *W, const vfft_config_t *cfg, struct vfft_plan_s *h,
                           int N1, int N2, int ord, int T, const char *val)
{
    if (!W || W->vw2_off_2d)
        return;
    if (vw2_2d_rl_tok_sets(&W->vw2, N1, N2, ord, T, "raxis", val) != 0)
    {
        vw2_2d_rl_bank(&W->vw2, N1, N2, 0, h->il2d_col.R, h->il2d_col.nst, -1, -1, 0,
                       (N1 & (N1 - 1)) ? h->il2d_col.blu : -1, 0.0, ord, T);
        if (vw2_2d_rl_tok_sets(&W->vw2, N1, N2, ord, T, "raxis", val) != 0)
        {
            fprintf(stderr, "vfft: the 2D real raxis verdict NOT banked at %dx%d -- the cell will re-race\n", N1, N2);
            return;
        }
    }
    _vw2_persist(W, cfg);
}

/* ── the race: the standard walk against the form, on the whole transform ── */
typedef struct
{
    struct vfft_plan_s *h;
    const double *x;
    double *z;
} _il2d_rax_arm_t;
static void _il2d_rax_arm_std(void *v)
{
    _il2d_rax_arm_t *a = (_il2d_rax_arm_t *)v;
    _il2d_real_rows_fwd(a->h, a->x, a->z);
    _il2d_real_cols(a->h, a->z, a->z, /*reverse=*/0);
}
static void _il2d_rax_arm_rax(void *v)
{
    _il2d_rax_arm_t *a = (_il2d_rax_arm_t *)v;
    _il2d_rax_exec_fwd(a->h, a->x, a->z);
}

/* THE FORM'S PLAN: env pin, the banked raxis=, or the race. Runs after the
 * standard walk's row and column plans are bound (the race runs them). */
static void _il2d_rax_plan(struct vfft_plan_s *h, struct vfft_wisdom_s *W, const vfft_config_t *cfg,
                           int N1, int N2, int ord, int T)
{
    const char *log = getenv("VFFT_IL2D_LOG"), *e = getenv("VFFT_IL2D_RAXIS");
    h->il2d_rax_on = 0;
    if (!_il2d_rax_admits(h, cfg))
        return;
    if (e && e[0])
    {   /* the pin: beats wisdom, never banks */
        if (!strcmp(e, "n2") || !strcmp(e, "0"))
            return;
        if (!strcmp(e, "n1") || !strcmp(e, "1"))
        {
            if (_il2d_rax_build(h, W, cfg, ord, T))
                h->il2d_rax_on = 1;
            else
            {
                _il2d_rax_free(h);
                _vfft_warn("vfft_create: VFFT_IL2D_RAXIS=n1 does not build at %dx%d (the standard walk stands)", N1, N2);
            }
            return;
        }
        _vfft_warn("vfft_create: VFFT_IL2D_RAXIS=%s is not n1 or n2 (the standard walk stands)", e);
        return;
    }
    if (!W || W->vw2_off_2d)
        return;
    if (!cfg->recalibrate)
    {   /* the banked verdict */
        const char *tok = vw2_2d_rl_tok_gets(&W->vw2, N1, N2, ord, T, "raxis");
        if (tok)
        {
            if (strcmp(tok, "n1"))
                return;   /* n2: the standard walk */
            if (_il2d_rax_build(h, W, cfg, ord, T))
            {
                h->il2d_rax_on = 1;
                return;
            }
            _il2d_rax_free(h);   /* a banked form that no longer builds: the race */
        }
    }
    if (!_il2d_rax_build(h, W, cfg, ord, T))
    {   /* nothing banked: the form may build another day */
        _il2d_rax_free(h);
        return;
    }
    {
        const size_t hp1 = (size_t)N2 / 2 + 1, RN = (size_t)N1 * (size_t)N2, CN = 2 * (size_t)N1 * hp1;
        double *x = (double *)vfft_aligned_alloc((RN + 8) * sizeof(double));
        double *z = (double *)vfft_aligned_alloc((CN + 8) * sizeof(double));
        double *zr = (double *)vfft_aligned_alloc((CN + 8) * sizeof(double));
        _il2d_rax_arm_t as, ar;
        vfft_race_arm_t arms[2];
        double ns[2], t0, est, err;
        int reps, win;
        if (!x || !z || !zr)
        {
            vfft_aligned_free(x); vfft_aligned_free(z); vfft_aligned_free(zr);
            _il2d_rax_free(h);
            return;
        }
        {
            unsigned sd = 0x9e3779b9u ^ (unsigned)N1 ^ ((unsigned)N2 << 12);
            size_t j;
            for (j = 0; j < RN + 8; j++)
            {
                sd = sd * 1664525u + 1013904223u;
                x[j] = (double)(sd >> 8) / (double)(1u << 24) - 0.5;
            }
        }
        as.h = h; as.x = x; as.z = zr;
        ar.h = h; ar.x = x; ar.z = z;
        /* the gate: the form's plane against the standard walk's */
        memset(zr, 0, (CN + 8) * sizeof(double));
        memset(z, 0, (CN + 8) * sizeof(double));
        _il2d_rax_arm_std(&as);
        _il2d_rax_arm_rax(&ar);
        err = _zrpr_relerr(z, zr, CN);
        if (!(err < 1e-10))
        {
            fprintf(stderr, "[il2d-real] raxis %dx%d: the real axis on N1 FAILS the gate (rel %.2e) -- dropped\n", N1, N2, err);
            vfft_aligned_free(x); vfft_aligned_free(z); vfft_aligned_free(zr);
            _il2d_rax_free(h);
            return;
        }
        arms[0].name = "n2"; arms[0].run = _il2d_rax_arm_std; arms[0].ctx = &as;
        arms[1].name = "n1"; arms[1].run = _il2d_rax_arm_rax; arms[1].ctx = &ar;
        _vfft_create_race_count++;
        t0 = vfft_now_ns();
        _il2d_rax_arm_std(&as);
        est = vfft_now_ns() - t0;
        reps = (int)(3.0e5 / (est > 1.0 ? est : 1.0));
        if (reps < 1) reps = 1;
        if (reps > 4096) reps = 4096;
        {   /* 9 rounds alternated, median, paced: the row plan race's protocol */
            const vfft_race_proto_t proto = { 9, reps, VFFT_RACE_MEDIAN, 1, 1, NULL, NULL, 1, 0 };
            vfft_race_run(&proto, arms, 2, ns);
        }
        win = vfft_race_beats(ns[1], ns[0], VFFT_RACE_HYST);   /* the race's margin toward the standard walk */
        if (log)
        {
            char cs[48];
            int i, off = 0;
            cs[0] = 0;
            for (i = 0; i < h->il2d_rax_col.nst && off < 40; i++)
                off += snprintf(cs + off, sizeof cs - (size_t)off, "%s%d", i ? "." : "", h->il2d_rax_col.R[i]);
            fprintf(stderr, "[il2d-real] raxis %dx%d race: reps=%d | n2 (rows then columns)=%.0f n1 (the real axis on N1: chain %s%s at M=%d)=%.0f -> %s\n",
                    N1, N2, reps, ns[0], cs, h->il2d_rax_col.blu ? " blu" : "", h->il2d_rax_M, ns[1], win ? "n1" : "n2");
        }
        vfft_aligned_free(x); vfft_aligned_free(z); vfft_aligned_free(zr);
        if (win)
            h->il2d_rax_on = 1;
        else
            _il2d_rax_free(h);
        _il2d_rax_bank(W, cfg, h, N1, N2, ord, T, win ? "n1" : "n2");
    }
}

#endif /* VFFT_IL2D_REAL_AXIS_H */
