/* vfft_batch.h — the OWNED-BATCH allocator (migration step 28).
 *
 * WHAT THIS IS
 * ------------
 * The eight functions behind `config.owned_buffers` and `config.batch`: plane
 * allocation, the three descriptor shapes, the padded-stride decision, and the
 * accessors the public API exposes them through. A closed cluster — everything
 * here is called either from within it or from vfft_create / vfft_destroy /
 * vfft_plan_planes / vfft_plan_stride.
 *
 * THE HANDLE IS Kp-WIDE, AND Kp IS THE POINT
 * ------------------------------------------
 * A padded batch is roundup(K, VW) lanes wide with the pad columns ZEROED, and
 * the plan is built at Kp and run at the padded wisdom's exec_me. Zeroing is
 * not hygiene: the pad lanes are transformed like any other, so non-zero pad
 * columns produce real numbers in lanes the caller never asked about.
 *
 * 🔴 VW IS HARDCODED TO 4 (AVX2 host). A single VW source-of-truth is needed
 * before an AVX-512 (VW=8) extension; see the padding design doc.
 *
 * THREE DESCRIPTOR SHAPES, NOT ONE
 * --------------------------------
 * They are genuinely different geometries and a mismatched handle is refused
 * rather than reinterpreted:
 *   c2c in-place   re/im, each N*Kp;
 *   real (r2c/c2r) a real plane of N*Kp against a spectrum of (N/2+1)*Kp;
 *   OOP            4-plane (re/im in, ore/oim out), Kp = roundup(K, 8).
 *
 * WHY THE REFUSALS LIVE HERE AND ARE LOUD
 * ---------------------------------------
 * `_own_batch_for` speaks in vfft_create's voice ("vfft_create(owned_buffers):
 * ...") because from the caller's side that IS vfft_create failing — the batch
 * is an implementation detail of the config they passed. Padded batches are 1D
 * only, split-plane by construction, and need even N for the real families;
 * each of those is a config-space mistake, so each is refused loudly rather
 * than by a bare NULL.
 *
 * PADDING IS RACED, NOT ASSUMED
 * -----------------------------
 * `_pad_stride_c2c` does not apply a rule of thumb. It consults the store and,
 * on a miss, calls `_calibrate_pad` — pad-vs-tail is a measured verdict per
 * cell (see planning/pad_calibrate.h). Nothing here may grow a hand cutoff.
 *
 * POSITION IN vfft.c IS LOAD-BEARING
 * ----------------------------------
 * Not a standalone header. It calls file-scope statics that live in vfft.c
 * (_calibrate_c2c, _calibrate_pad, _default_wisdom, _registry, _vw2_persist),
 * so it must be included after those are defined and before vfft_destroy — the
 * one caller that sits ABOVE the cluster's original position.
 */
#ifndef VFFT_BATCH_H
#define VFFT_BATCH_H

/* ── padded batch: Kp=roundup(K,VW)-wide, ZEROED re+im, opaque handle ──
 * VW hardcoded to 4 (AVX2 host). A single VW source-of-truth is needed before an
 * AVX-512 (VW=8) extension — see the padding design doc. This handle DRIVES the padded
 * c2c in-place execute path: pass it as config.batch to vfft_create and the plan is built
 * at Kp + run at the padded wisdom's exec_me (see the padded branch in vfft_create). */
/* stride_alloc + zero (pad columns MUST be zero); NULL-safe caller frees on partial fail. */
static double *_batch_plane(size_t doubles)
{
    double *p = (double *)stride_alloc(doubles * sizeof(double));
    if (p)
        memset(p, 0, doubles * sizeof(double)); /* stride_alloc does NOT zero */
    return p;
}

/* Internal transform-aware padded allocator (validation lives in the public
 * _own_batch_for door). c2c is in-place (re/im only, N*Kp each); r2c/c2r
 * are out-of-place with a real plane (N*Kp) and a split spectrum ((N/2+1)*Kp each); trig
 * (DCT/DST/DHT) is real->real out-of-place: real = INPUT plane, re = OUTPUT plane (both
 * N*Kp), im unused. All planes Kp-strided so the Kp-built plan lands exactly (element e,
 * lane t -> [e*Kp+t]). */
/* Kp_forced: 0 = use the default roundup; else the caller's measured stride
 * (see _pad_stride_c2c — K means "allocate tight, padding lost the race"). */
static vfft_batch _batch_alloc_ex(vfft_transform_t xform, int N, size_t K,
                                  size_t Kp_forced)
{
    int real_side = (xform == VFFT_R2C || xform == VFFT_C2R);
    int trig = _VFFT_IS_TRIG(xform);
    size_t Kp = Kp_forced ? Kp_forced : ((K + 3u) & ~(size_t)3u); /* roundup(K, VW=4) */
    struct vfft_batch_s *b = (struct vfft_batch_s *)calloc(1, sizeof *b);
    if (!b)
        return NULL;
    b->N = N;
    b->K = K;
    b->Kp = Kp;
    b->xform = (int)xform;
    int ok = 1;
    if (trig) /* real -> real, out-of-place: input plane + output plane, both N*Kp */
    {
        size_t data = (size_t)N * Kp;
        b->real = _batch_plane(data); /* INPUT plane */
        b->re = _batch_plane(data);   /* OUTPUT plane */
        ok = (b->real && b->re);
    }
    else if (real_side)
    {
        size_t spec = (size_t)(N / 2 + 1) * Kp; /* split spectrum plane */
        b->real = _batch_plane((size_t)N * Kp); /* real plane */
        b->re = _batch_plane(spec);
        b->im = _batch_plane(spec);
        ok = (b->real && b->re && b->im);
    }
    else /* c2c in-place: split data, no real plane */
    {
        size_t data = (size_t)N * Kp;
        b->re = _batch_plane(data);
        b->im = _batch_plane(data);
        ok = (b->re && b->im);
    }
    if (!ok)
    {
        _own_batch_free(b);
        return NULL;
    }
    return b;
}
/* Internal OOP c2c padded handle: 4 split planes (re/im INPUT, ore/oim OUTPUT), each N*Kp.
 * Kp = roundup(K,8) (NOT VW=4): OOP BAILEY2 hard-gates on K%8 (oop_auto.h) and the OOP wisdom
 * READER rejects K%8!=0 (oop_wisdom.h) — an 8-aligned Kp keeps all 3 kinds AND lets the
 * (N,Kp) plan cache, with zero changes to the OOP internals. */
static vfft_batch _batch_alloc_oop(int N, size_t K)
{
    size_t Kp = (K + 7u) & ~(size_t)7u; /* roundup(K, 8) — OOP kind + wisdom alignment */
    struct vfft_batch_s *b = (struct vfft_batch_s *)calloc(1, sizeof *b);
    if (!b)
        return NULL;
    b->N = N;
    b->K = K;
    b->Kp = Kp;
    b->xform = (int)VFFT_C2C;
    b->oop = 1;
    size_t data = (size_t)N * Kp;
    b->re = _batch_plane(data); /* INPUT re/im */
    b->im = _batch_plane(data);
    b->ore = _batch_plane(data); /* OUTPUT re/im */
    b->oim = _batch_plane(data);
    if (!(b->re && b->im && b->ore && b->oim))
    {
        _own_batch_free(b);
        return NULL;
    }
    return b;
}

static void _own_batch_free(vfft_batch b)
{
    if (!b)
        return;
    if (b->real)
        stride_free(b->real); /* Windows: stride_free == _aligned_free; free() is UB */
    if (b->re)
        stride_free(b->re);
    if (b->im)
        stride_free(b->im);
    if (b->ore)
        stride_free(b->ore);
    if (b->oim)
        stride_free(b->oim);
    free(b);
}
/* THE public allocator (batch API consolidation, 9 fns -> 4): the batch is
 * BORN FROM THE CONFIG, so create's handle-vs-descriptor cross-check is total
 * by construction and the Kp rule (VW=4 tight vs OOP 8) is an internal detail
 * keyed off placement. Loud rejection on every unsupported combination —
 * same voice as vfft_create. */
/* Returns the stride to ALLOCATE at: K (tight won) or Kp (padded won). Wisdom
 * hit is instant; a miss races _calibrate_pad and banks only a nonzero verdict.
 * See docs/design/vfft_front_door.md. */
static size_t _pad_stride_c2c(int N, size_t K, const vfft_config_t *cfg)
{
    const size_t Kp = (K + (size_t)(_VFFT_PADVW - 1)) & ~(size_t)(_VFFT_PADVW - 1);
    if (Kp == K)
        return K; /* already lane-aligned: nothing to decide */
    if (_vfft_is_prime(N))
        return K; /* no CT factorization to race (prime runs its own engine) */

    struct vfft_wisdom_s *W = cfg->wisdom ? cfg->wisdom : _default_wisdom();
    const vfft_proto_registry_t *reg = _registry();
    const vfft_proto_wisdom_entry_t *te = vfft_proto_wisdom_lookup(&W->c2c, N, K);
    /* wave-4: seed the process cache from the STORE (both legs) */
    if (!W->vw2_off_stride)
    {
        /* store-hit OVERWRITES the (possibly stale) frozen-file preload */
        vfft_proto_wisdom_entry_t sb;
        if (vw2_stride_lookup(&W->vw2, 0, N, K, &sb))
            vfft_proto_wisdom_set(&W->c2c, &sb);
        if (vw2_stride_lookup(&W->vw2, 0, N, Kp, &sb))
            vfft_proto_wisdom_set(&W->c2c, &sb);
        te = vfft_proto_wisdom_lookup(&W->c2c, N, K);
    }
    if (te && !cfg->recalibrate)
    {
        if (te->exec_me == (int)K)
            return K;
        if (te->exec_me == (int)Kp)
            return Kp;
    }
    /* MISS (or recalibrate): ensure both factorizations exist, then race. */
    int dirty = 0;
    if (!te || cfg->recalibrate)
    {
        vfft_proto_wisdom_entry_t ne;
        if (_calibrate_c2c(N, K, cfg->rigor, reg, &ne) == 0)
        {
            vfft_proto_wisdom_add(&W->c2c, &ne, 1);
            vw2_stride_bank_entry(&W->vw2, &ne, 0);
            dirty = 1;
        }
    }
    const vfft_proto_wisdom_entry_t *ae = vfft_proto_wisdom_lookup(&W->c2c, N, Kp);
    if (!ae || cfg->recalibrate)
    {
        vfft_proto_wisdom_entry_t ne;
        if (_calibrate_c2c(N, (size_t)Kp, cfg->rigor, reg, &ne) == 0)
        {
            vfft_proto_wisdom_add(&W->c2c, &ne, 1);
            vw2_stride_bank_entry(&W->vw2, &ne, 0);
            dirty = 1;
        }
    }
    te = vfft_proto_wisdom_lookup(&W->c2c, N, K); /* wisdom_add may realloc */
    ae = vfft_proto_wisdom_lookup(&W->c2c, N, Kp);
    size_t stride = K; /* fall back to tight (the drop-in default) */
    if (te && ae)
    {
        int verdict = _calibrate_pad(N, K, cfg->rigor, reg, te, ae); /* Kp / K / 0 */
        if (verdict > 0)
        {
            vfft_proto_wisdom_entry_t upd = *te; /* keep factK, stamp the verdict */
            upd.exec_me = verdict;
            vfft_proto_wisdom_add(&W->c2c, &upd, 1);
            vw2_stride_bank_entry(&W->vw2, &upd, 0); /* pad_me= rides the record */
            dirty = 1;
            stride = (size_t)verdict;
        }
    }
    if (dirty)
        _vw2_persist(W, cfg);
    return stride;
}

static vfft_batch _own_batch_for(const vfft_config_t *cfg)
{
    if (!cfg)
    {
        _vfft_warn("vfft_create(owned_buffers): NULL config");
        return NULL;
    }
    if ((int)cfg->transform < (int)VFFT_C2C || (int)cfg->transform > (int)VFFT_DHT)
    {
        _vfft_warn("vfft_create(owned_buffers): invalid transform enum %d (valid: VFFT_C2C..VFFT_DHT)",
                   (int)cfg->transform);
        return NULL;
    }
    if (cfg->dims > 1)
    {
        _vfft_warn("vfft_create(owned_buffers): padded batches are 1D only (got dims=%d) — "
                   "2D+ plans run tight buffers",
                   cfg->dims);
        return NULL;
    }
    if ((int)cfg->layout == (int)VFFT_LAYOUT_INTERLEAVED)
    {
        _vfft_warn("vfft_create(owned_buffers): padded batches are split-plane by construction — "
                   "layout must be VFFT_LAYOUT_SPLIT");
        return NULL;
    }
    if (cfg->n[0] < 1 || cfg->howmany < 1)
    {
        _vfft_warn("vfft_create(owned_buffers): need n[0] >= 1 and howmany >= 1 (got N=%d K=%zu)",
                   cfg->n[0], cfg->howmany);
        return NULL;
    }
    int N = cfg->n[0];
    size_t K = cfg->howmany;
    if (cfg->transform == VFFT_C2C)
    {
        if (cfg->placement == VFFT_OUTOFPLACE)
            return _batch_alloc_oop(N, K); /* 4-plane, Kp=roundup(K,8) — the OOP
                                            * kinds hard-gate on K%8, so padding
                                            * is structural there, not a verdict */
        /* in-place: the MEASURED tight-vs-padded verdict picks the stride */
        return _batch_alloc_ex(VFFT_C2C, N, K, _pad_stride_c2c(N, K, cfg));
    }
    if ((cfg->transform == VFFT_R2C || cfg->transform == VFFT_C2R ||
         _VFFT_IS_TRIG(cfg->transform)) &&
        (N % 2) != 0)
    {
        _vfft_warn("vfft_create(owned_buffers): %s padding requires even N (real-FFT inner "
                   "half-spectrum); got N=%d",
                   _vfft_tname(cfg->transform), N);
        return NULL;
    }
    /* real/trig: pad-only by construction (padding is their ONLY full-SIMD path
     * for misaligned K), so they keep the roundup default — no verdict applies. */
    return _batch_alloc_ex(cfg->transform, N, K, 0);
}

/* Fill the vfft_execute arguments in the right roles — the role table lives
 * HERE, once, inside the library (see the vfft.h padded-batch section).
 * Roles are the forward data flow; slots the transform does not use are set
 * to NULL; out-params may themselves be NULL if unwanted. */
static void _own_batch_planes(vfft_batch b, double **sre, double **sim,
                              double **dre, double **dim)
{
    double *s_re = NULL, *s_im = NULL, *d_re = NULL, *d_im = NULL;
    if (!b)
        _vfft_warn("vfft_plan_planes: NULL batch handle — all planes set to NULL");
    else if (b->oop)
    { /* c2c out-of-place: input planes -> src, output planes -> dst */
        s_re = b->re;
        s_im = b->im;
        d_re = b->ore;
        d_im = b->oim;
    }
    else if (b->xform == (int)VFFT_C2C)
    { /* c2c in-place: the same planes fill both roles */
        s_re = b->re;
        s_im = b->im;
        d_re = b->re;
        d_im = b->im;
    }
    else if (b->xform == (int)VFFT_R2C)
    { /* real in -> split spectrum out */
        s_re = b->real;
        d_re = b->re;
        d_im = b->im;
    }
    else if (b->xform == (int)VFFT_C2R)
    { /* split spectrum in -> real out */
        s_re = b->re;
        s_im = b->im;
        d_re = b->real;
    }
    else
    { /* trig: real in -> real out */
        s_re = b->real;
        d_re = b->re;
    }
    if (sre)
        *sre = s_re;
    if (sim)
        *sim = s_im;
    if (dre)
        *dre = d_re;
    if (dim)
        *dim = d_im;
}

static size_t _own_batch_stride(vfft_batch b) { return b ? b->Kp : 0; }

#endif /* VFFT_BATCH_H */
