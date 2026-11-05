/* real_create_il.h — the r2c / c2r CREATE, interleaved tier.
 *
 * K==1, INTERLEAVED, no caller-supplied batch. The door's engine pick is by
 * the parity of N:
 *   even  il/real/zrp_build.h: zr2c (the CCE plane reinterpreted as z[N/2],
 *         a c2c child, and the Hermitian fold), zrp (the real pair), zttr
 *         (ZTT-r), zfsr (the real four-step) and zrm (the real mono);
 *   odd   il/real/odd_build.h: zrm (the real mono), zrf (the real flat DIT)
 *         and zrb (the real Bluestein), both placements.
 * The cell's engine is banked in the real shard (wisdom2_real_il.h); a miss
 * races. The door below (_vfft_create_real_door) holds the gate and calls
 * this only for a matching request.
 *
 * Returns the handle, or NULL when no engine could be built at N -- never
 * another library's engine. *refused=1 says this door already warned (an
 * in-place request, any odd request); *refused=0 leaves the warning to the
 * door, which refuses an even out-of-place cell the same way.
 *
 * It runs BEFORE the split-path calibrate-on-miss blocks on purpose: a
 * zr2c-served cell must not pay for (or bank) c2c(N/2, K)/rfft rows it never
 * reads — the child rides the K=1 engine tables through its own recursive
 * create.
 *
 * POSITION IN vfft.c IS LOAD-BEARING: calls file-scope statics of vfft.c, so
 * it is included (via the dispatcher) after those are defined.
 */
#ifndef VFFT_IL_REAL_CREATE_IL_H
#define VFFT_IL_REAL_CREATE_IL_H

static struct vfft_plan_s *_vfft_create_real_il(const vfft_cfg_t *cfg,
                                                struct vfft_wisdom_s *W,
                                                int N, int *refused)
{
    struct vfft_plan_s *hz = (N & 1) ? _real_il_odd_build(cfg, N, W) : _real_il_build(cfg, N, W);
    *refused = 0;
    if (hz)
        return hz; /* the banked engine, or the race's winner: banks its own cell */
    if (N & 1)
    {
        _vfft_warn("vfft_create: %s odd N=%d %s: no IL real engine built at this cell "
                   "(every arm failed to build or was gated out); unsupported",
                   _vfft_tname(cfg->transform), N,
                   cfg->placement == VFFT_INPLACE ? "in place" : "out of place");
        *refused = 1;
        return NULL;
    }
    /* 🔴 NO SILENT DEGRADE TO OUT-OF-PLACE. The in-place refusal in create
     * ADMITTED this shape, so falling through would stamp
     * h->placement = INPLACE onto a handle whose executor is the OOP
     * CCE path -- engines that stream an N-double real plane into an
     * N+2-double CCE plane and were never gated for aliasing. The
     * caller then makes the documented (z,NULL,z,NULL) call and gets
     * an out-of-place executor whose source aliases its destination.
     * The interleaved engines are the ONLY in-place real path, so if none
     * could be built there is no in-place plan to give: refuse loudly.
     * Out-of-place callers keep the fall-through unchanged. */
    if (cfg->placement == VFFT_INPLACE)
    {
        _vfft_warn("vfft_create: in-place %s N=%d could not build the zr2c route "
                   "(the only in-place real path); no out-of-place fallback exists "
                   "for an in-place plan -- use VFFT_PLACEMENT_OUT",
                   _vfft_tname(cfg->transform), N);
        *refused = 1;
    }
    return NULL;
}

/* ── the door's ONE exit: the bound 1D real dispatch (il/il_execute.h) ── */
static vfft_plan _vfft_real_bind_exec(vfft_plan hp);
static vfft_plan _real_finish(struct vfft_plan_s *h)
{
    return _vfft_real_bind_exec((vfft_plan)h);
}

/* THE LANE BLUESTEIN'S CREATE (il/real/zrb_lanes.h): the interleaved real
 * batch in its lane-major geometry, K > 1, out of place, at an odd N without
 * a chain. The plan inputs -- the length M, the column chain at M with its
 * forms, the column window -- are swept at create: the column pool's lengths
 * at or above the bound (the smallest power of two, the three smallest
 * smooth lengths, the two smallest with a chain), every chain the enumerator
 * yields at each (the no-silent-caps law is the enumerator's), the window
 * ladder on the two fastest; every candidate gated against K one-lane
 * transforms through the doors' independent reference and burst-timed; the two fastest race
 * (the library body, paced) and the winner is banked on the cell's q=K row.
 * VFFT_ZRBL=0 keeps it out; NULL = nothing built (the request goes where it
 * went). */
static struct vfft_plan_s *_zrbl_build_plan(const vfft_cfg_t *cfg, int N, int K, int M,
                                            const int *Rs, int nst, const char *forms, int wc)
{
    vfft_zrbl_plan_t *zp = vfft_zrbl_create(N, K, M, Rs, nst, forms, wc);
    struct vfft_plan_s *h;
    if (!zp)
        return NULL;
    h = (struct vfft_plan_s *)calloc(1, sizeof *h);
    if (!h)
    {
        vfft_zrbl_destroy(zp);
        return NULL;
    }
    h->transform = cfg->transform;
    h->placement = VFFT_OUTOFPLACE;
    h->layout = (int)VFI_LAYOUT_IL;
    h->N = N;
    h->K = K;
    h->nthreads = _vfft_plan_threads(cfg);
    h->zrbl = zp;
    return h;
}
static int _zrbl_nchains(int M, int (*ch)[8], int *lens, int warn)
{
    int cur[8], n = 0, dropped = 0;
    memset(cur, 0, sizeof cur);
    if (warn) _il2d_enum_rec(M, 0, cur, ch, lens, &n, &dropped);
    else _il2d_enum_rec_body(M, 0, cur, ch, lens, &n, &dropped);
    return n;
}
static int _zrbl_m_cands(int N, int *out)
{
    static const int q[5] = { 3, 5, 7, 9, 15 };
    static int ch[VFFT_IL2D_POOL_MAX][8], lens[VFFT_IL2D_POOL_MAX];
    const int mn = vfft_zrb_min_m(N);
    int sm[64], nsm = 0, n = 0, p2 = 16, i, a, k, M;
    while (p2 < mn) p2 <<= 1;
    for (i = 0; i < 5; i++)
        for (a = 2; (1 << a) * q[i] < p2 && nsm < 64; a++)
            if ((1 << a) * q[i] >= mn) sm[nsm++] = (1 << a) * q[i];
    for (i = 1; i < nsm; i++)
        for (k = i; k > 0 && sm[k] < sm[k - 1]; k--) { const int t = sm[k]; sm[k] = sm[k - 1]; sm[k - 1] = t; }
    for (i = 0, k = 0; i < nsm && k < 3; i++)
        if (_zrbl_nchains(sm[i], ch, lens, 0) > 0) { out[n++] = sm[i]; k++; }
    for (M = mn, k = 0; M < p2 && M <= 65536 && k < 2; M++)
    {
        int dup = 0;
        for (i = 0; i < n; i++) if (out[i] == M) dup = 1;
        if (dup) continue;
        if (_zrbl_nchains(M, ch, lens, 0) > 0) { out[n++] = M; k++; }
    }
    out[n++] = p2;
    return n;
}
static int _zrbl_try(const vfft_cfg_t *cfg, int N, int K, int M, const int *Rs, int nst, const char *forms, int wc,
                     const double *a, const double *ref, double *b, size_t nout,
                     struct vfft_plan_s *out[2], double bns[2])
{
    struct vfft_plan_s *h = _zrbl_build_plan(cfg, N, K, M, Rs, nst, forms, wc);
    double t0, est, best = 1e300;
    int reps;
    if (!h)
        return 0;
    memset(b, 0, nout * sizeof(double));
    _exec_zrbl(h, a, b);
    {
        const double e = _zrpr_relerr(b, ref, nout);
        if (e >= 1e-10)
        {
            char cs[128];
            vfft_zrbl_str(h->zrbl, cs, sizeof cs);
            fprintf(stderr, "[zrbl] N=%d K=%d %s FAILS the gate (rel %.2e vs the reference) -- dropped\n", N, K, cs, e);
            vfft_destroy((vfft_plan)h);
            return 1;
        }
    }
    _exec_zrbl(h, a, b);
    t0 = vfft_now_ns();
    _exec_zrbl(h, a, b);
    est = vfft_now_ns() - t0;
    reps = (int)(1.0e5 / (est > 1.0 ? est : 1.0));
    if (reps < 2) reps = 2;
    if (reps > 256) reps = 256;
    for (int r = 0; r < 3; r++)
    {
        double t = vfft_now_ns();
        for (int i = 0; i < reps; i++) _exec_zrbl(h, a, b);
        t = (vfft_now_ns() - t) / reps;
        if (t < best) best = t;
    }
    if (best < bns[0])
    {
        if (out[1]) vfft_destroy((vfft_plan)out[1]);
        out[1] = out[0]; bns[1] = bns[0];
        out[0] = h; bns[0] = best;
    }
    else if (best < bns[1])
    {
        if (out[1]) vfft_destroy((vfft_plan)out[1]);
        out[1] = h; bns[1] = best;
    }
    else
        vfft_destroy((vfft_plan)h);
    return 1;
}
static int _zrbl_sweep(const vfft_cfg_t *cfg, int N, int K, const double *a, const double *ref, double *b,
                       size_t nout, struct vfft_plan_s *out[2])
{
    static const int wins[6] = { 2, 4, 8, 16, 32, 64 };
    static int ch[VFFT_IL2D_POOL_MAX][8], lens[VFFT_IL2D_POOL_MAX];
    int Ms[VFFT_ZRB_MAX_M + 2];
    const int nm = _zrbl_m_cands(N, Ms);
    double bns[2] = { 1e300, 1e300 };
    int sR[2][8], sn[2] = { 0, 0 }, sM[2] = { 0, 0 };
    char sf[2][64];
    out[0] = out[1] = NULL;
    for (int m = 0; m < nm; m++)
    {
        const int nc = _zrbl_nchains(Ms[m], ch, lens, 1);
        int nb = 0;
        for (int c = 0; c < nc; c++)
            nb += _zrbl_try(cfg, N, K, Ms[m], ch[c], lens[c], "", 0, a, ref, b, nout, out, bns);
        if (getenv("VFFT_ZRACE_VERBOSE"))
            fprintf(stderr, "[zrbl] N=%d K=%d M=%d: %d of %d chain(s) built; best so far %.0f ns\n", N, K, Ms[m], nb, nc, bns[0]);
    }
    for (int i = 0; i < 2; i++)
        if (out[i])
        {
            sn[i] = out[i]->zrbl->nst; sM[i] = out[i]->zrbl->M;
            memcpy(sR[i], out[i]->zrbl->Rs, sizeof(int) * (size_t)sn[i]);
            snprintf(sf[i], sizeof sf[i], "%s", out[i]->zrbl->forms);
        }
    for (int i = 0; i < 2; i++)
        for (int w = 0; sn[i] && w < 6 && wins[w] < K; w++)
            _zrbl_try(cfg, N, K, sM[i], sR[i], sn[i], sf[i], wins[w], a, ref, b, nout, out, bns);
    return (out[0] != NULL) + (out[1] != NULL);
}
static vfft_plan _vfft_create_real_lanes(const vfft_cfg_t *cfg, struct vfft_wisdom_s *W, int N, int K)
{
    const int c2r = cfg->transform == VFFT_C2R;
    const int Tk = _vfft_plan_threads(cfg);
    const size_t hp1 = (size_t)N / 2 + 1, nin = c2r ? 2 * hp1 * (size_t)K : (size_t)N * (size_t)K;
    const size_t nout = c2r ? (size_t)N * (size_t)K : 2 * hp1 * (size_t)K;
    const char *e = getenv("VFFT_ZRBL");
    struct vfft_plan_s *hz[2] = { NULL, NULL };
    double *a, *ref, *b, *ti, *to;
    int refbad = 0;
    vfft_cfg_t c1;
    if (e && e[0] == '0' && !e[1])
        return NULL;
    if (W && !W->vw2_off_oop && !cfg->recalibrate)
    {
        int M = 0, Rs[8], nst = 0, wc = 0, tw = 0;
        char forms[64], kind[8], shape[64];
        _ilprime_inner_desc_t d;
        if (vw2_real_il_lookup_zrbl(&W->vw2, N, K, c2r, 0, Tk, &M, Rs, 8, &nst, forms, sizeof forms, &wc))
        {
            struct vfft_plan_s *h = _zrbl_build_plan(cfg, N, K, M, Rs, nst, forms, wc);
            if (h)
                return _real_finish(h);
        }
        if (vw2_real_il_lookup_zrb_q(&W->vw2, N, K, c2r, 0, Tk, &M, kind, sizeof kind, shape, sizeof shape, &tw, NULL) &&
            _ilprime_desc_parse(&d, kind, shape, tw))
        {   /* the one-row engine over the lanes, its edges at the lane stride */
            struct vfft_plan_s *h = _zrb_build_plan(cfg, N, M, &d);
            if (h)
            {
                h->K = K;
                return _real_finish(h);
            }
        }
    }
    if (!W || W->vw2_off_oop)
        return NULL;
    /* the reference: K one-lane transforms through the doors' independent
     * reference (_real_il_ref, il/real/zrp_build.h) */
    c1 = *cfg;
    c1.howmany = 1;
    c1.batch_geom = VFFT_BATCH_DEFAULT;
    c1.placement = VFFT_OUTOFPLACE;
    a = (double *)vfft_aligned_alloc((nin + 8) * sizeof(double));
    ref = (double *)vfft_aligned_alloc((nout + 8) * sizeof(double));
    b = (double *)vfft_aligned_alloc((nout + 8) * sizeof(double));
    ti = (double *)vfft_aligned_alloc((2 * hp1 + 8) * sizeof(double));
    to = (double *)vfft_aligned_alloc((2 * hp1 + 8) * sizeof(double));
    if (!a || !ref || !b || !ti || !to)
    {
        vfft_aligned_free(a); vfft_aligned_free(ref); vfft_aligned_free(b); vfft_aligned_free(ti); vfft_aligned_free(to);
        return NULL;
    }
    {
        unsigned sd = 0x9e3779b9u ^ (unsigned)N ^ (unsigned)(K << 12) ^ (unsigned)(c2r << 8);
        for (size_t i = 0; i < nin; i++)
        {
            sd = sd * 1664525u + 1013904223u;
            a[i] = (double)(sd >> 8) / (double)(1u << 24) - 0.5;
        }
        if (c2r)
            for (int t = 0; t < K; t++) a[2 * t + 1] = 0.0;   /* a CCE spectrum: real DC in every lane */
        memset(ref, 0, nout * sizeof(double));
        for (int t = 0; t < K && !refbad; t++)
        {   /* lane t out, through the one-lane reference, back into its lane */
            if (c2r)
            {
                for (size_t f = 0; f < hp1; f++) { ti[2 * f] = a[2 * (f * (size_t)K + t)]; ti[2 * f + 1] = a[2 * (f * (size_t)K + t) + 1]; }
                if (_real_il_ref(1, N, ti, to) != 0) { refbad = 1; break; }
                for (size_t n = 0; n < (size_t)N; n++) ref[n * (size_t)K + t] = to[n];
            }
            else
            {
                for (size_t n = 0; n < (size_t)N; n++) ti[n] = a[n * (size_t)K + t];
                if (_real_il_ref(0, N, ti, to) != 0) { refbad = 1; break; }
                for (size_t f = 0; f < hp1; f++) { ref[2 * (f * (size_t)K + t)] = to[2 * f]; ref[2 * (f * (size_t)K + t) + 1] = to[2 * f + 1]; }
            }
        }
    }
    vfft_aligned_free(ti); vfft_aligned_free(to);
    if (refbad)
    {
        _vfft_warn("vfft_create: %s N=%d K=%d lane-major: the gate's reference failed its self-check; no engine is raced",
                   _vfft_tname(cfg->transform), N, K);
        vfft_aligned_free(a); vfft_aligned_free(ref); vfft_aligned_free(b);
        return NULL;
    }
    {
        /* THE ARMS: the column form's podium (two), and the one-row engine
         * over the lanes with its edges at the lane stride -- built from the
         * cell's one-row verdict (raced now when the q=1 row is a miss) */
        struct vfft_plan_s *arm[3];
        double ns[3];
        int narm = 0, win = 0;
        if (_zrbl_sweep(cfg, N, K, a, ref, b, nout, hz) > 0)
        {
            arm[narm++] = hz[0];
            if (hz[1]) arm[narm++] = hz[1];
        }
        {
            struct vfft_plan_s *h1 = _real_il_odd_build(&c1, N, W); /* the cell's one-row verdict (il/real/odd_build.h) */
            if (h1 && h1->zrb)
            {
                _ilprime_inner_desc_t d;
                if (_ilprime_desc_parse(&d, h1->zrb->ikind, h1->zrb->ishape, h1->zrb->itw))
                {
                    struct vfft_plan_s *hk = _zrb_build_plan(cfg, N, h1->zrb->M, &d);
                    if (hk)
                    {
                        hk->K = K;
                        memset(b, 0, nout * sizeof(double));
                        _exec_zrb(hk, a, b);
                        if (_zrpr_relerr(b, ref, nout) < 1e-10)
                            arm[narm++] = hk;
                        else
                        {
                            fprintf(stderr, "[zrb] N=%d K=%d lanes FAIL the gate -- dropped\n", N, K);
                            vfft_destroy((vfft_plan)hk);
                        }
                    }
                }
            }
            if (h1) vfft_destroy((vfft_plan)h1);
        }
        if (narm == 0)
        {
            vfft_aligned_free(a); vfft_aligned_free(ref); vfft_aligned_free(b);
            _vfft_warn("vfft_create: %s N=%d K=%d lane-major: no IL batch engine built at this cell -- the request goes to the routes",
                       _vfft_tname(cfg->transform), N, K);
            return NULL;
        }
        _vfft_create_race_count++;
        if (narm > 1)
        {
            _odd_arm_t ca[3];
            vfft_race_arm_t arms[3];
            double t0, est;
            int reps;
            for (int i = 0; i < narm; i++)
            {
                ca[i].h = arm[i]; ca[i].in = a; ca[i].out = b;
                arms[i].name = arm[i]->zrbl ? (i ? "zrbl2" : "zrbl") : "zrbK";
                arms[i].run = _odd_arm_run; arms[i].ctx = &ca[i];
            }
            t0 = vfft_now_ns();
            _odd_arm_run(&ca[0]);
            est = vfft_now_ns() - t0;
            reps = (int)(3.0e5 / (est > 1.0 ? est : 1.0));
            if (reps < 2) reps = 2;
            if (reps > 1024) reps = 1024;
            {
                const vfft_race_proto_t proto = { 9, reps, VFFT_RACE_MEDIAN, 1, 1, NULL, NULL, 1 };
                vfft_race_run(&proto, arms, narm, ns);
            }
            for (int i = 1; i < narm; i++)
                if (ns[i] < ns[win]) win = i;
        }
        else
            ns[0] = 0.0;
        if (getenv("VFFT_ZRACE_VERBOSE"))
        {
            fprintf(stderr, "[lanes] N=%d K=%d %s race:", N, K, c2r ? "c2r" : "r2c");
            for (int i = 0; i < narm; i++)
            {
                char cs[128];
                if (arm[i]->zrbl) vfft_zrbl_str(arm[i]->zrbl, cs, sizeof cs);
                else vfft_zrb_str(arm[i]->zrb, cs, sizeof cs);
                fprintf(stderr, " %s:%s=%.0f%s", arm[i]->zrbl ? "zrbl" : "zrbK", cs, ns[i], i == win ? "*" : "");
            }
            fprintf(stderr, "\n");
        }
        vfft_aligned_free(a); vfft_aligned_free(ref); vfft_aligned_free(b);
        {
            struct vfft_plan_s *hw = arm[win];
            const int rc = hw->zrbl
                ? vw2_real_il_bank_zrbl(&W->vw2, N, K, c2r, 0, Tk, hw->zrbl->M, hw->zrbl->Rs, hw->zrbl->nst,
                                        hw->zrbl->forms, hw->zrbl->wc, ns[win])
                : vw2_real_il_bank_zrb_q(&W->vw2, N, K, c2r, 0, Tk, hw->zrb->M, hw->zrb->ikind, hw->zrb->ishape,
                                         hw->zrb->itw, 0, ns[win]);
            if (rc == VW2_OK)
                _vw2_persist(W, cfg);
            else
                fprintf(stderr, "vfft: lane verdict NOT banked at N=%d K=%d (rc=%d) -- the cell will re-race\n", N, K, rc);
            for (int i = 0; i < narm; i++)
                if (i != win) vfft_destroy((vfft_plan)arm[i]);
            return _real_finish(hw);
        }
    }
}

/* ── THE 1D REAL DOOR (2026-11-06: the bridge to the split real engines is
 * deleted; the split library is C2C only) ──────────────────────────────────
 * A 1D INTERLEAVED real request at K == 1: the odd door (the mono, the flat
 * DIT, the Bluestein) or the even door (zr2c, the pair, ZTT-r, the four-step,
 * the mono), each serving or refusing in the request's name. A batch never
 * reaches here: the transform-contiguous wrapper (vfft.c) runs K of these,
 * the lane-major batch has its own door (il/rank1/lm_batch_il.h) or is
 * refused before this with its reason (no native interleaved engine serves
 * the real lane-major geometry yet; the lane Bluestein's create above is the
 * odd-N piece of that engine, not reached today). */
static vfft_plan _vfft_create_real_door(const vfft_cfg_t *cfg, struct vfft_wisdom_s *W, int N, size_t K)
{
    int refused = 0;
    struct vfft_plan_s *h;
    if (K > 1)
    {
        _vfft_warn("vfft_create: %s N=%d howmany=%zu: no native interleaved engine serves the real "
                   "lane-major geometry yet -- name batch=VFFT_BATCH_CONTIGUOUS",
                   _vfft_tname(cfg->transform), N, K);
        return NULL;
    }
    if (_real_il_odd_admits(N, cfg->transform == VFFT_C2R))
    {   /* an odd cell with an IL real engine: the odd door serves it or refuses */
        h = _vfft_create_real_il(cfg, W, N, &refused);
        return h ? _real_finish(h) : NULL;
    }
    if (N & 1)
    {   /* an odd cell no IL real engine admits (the mono to 64, the flat chain,
         * the real Bluestein within its inner ceiling (3N-1)/2 <= VFFT_POLICY_INNER_MAX_N) */
        _vfft_warn("vfft_create: %s odd N=%d %s: no IL real engine serves this cell (no mono "
                   "kernel at N, no flat chain, and the real Bluestein's inner (3N-1)/2 = %ld "
                   "passes the %d inner ceiling); unsupported",
                   _vfft_tname(cfg->transform), N,
                   cfg->placement == VFFT_INPLACE ? "in place" : "out of place",
                   (3L * N - 1) / 2, VFFT_POLICY_INNER_MAX_N);
        return NULL;
    }
    /* an even cell: the even door serves it or refuses */
    h = _vfft_create_real_il(cfg, W, N, &refused);
    if (h)
        return _real_finish(h);
    if (!refused)
        _vfft_warn("vfft_create: %s N=%d out of place: no IL real engine built at this cell; unsupported",
                   _vfft_tname(cfg->transform), N);
    return NULL;
}

#endif /* VFFT_IL_REAL_CREATE_IL_H */
