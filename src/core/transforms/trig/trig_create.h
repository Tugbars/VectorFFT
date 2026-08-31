/* trig_create.h — the trig CREATE tier and its builders (migration step 27).
 *
 * WHAT THIS IS
 * ------------
 * The whole trig family's create path: DCT-I..IV, DST-I..III and DHT. Real in,
 * real out, on a real-FFT inner. Two things live here, and they are one unit
 * because nothing else calls into them:
 *
 *   the BUILDER CLUSTER — _vw2_t_of_trig, _inner_c2c_trig,
 *   _trig_r2c_set_inner_jit, _build_trig. A closed chain: each is called only
 *   by the next, and _build_trig only by the create tier below it.
 *
 *   the CREATE TIER — the _VFFT_IS_TRIG arm of _vfft_create_inner.
 *
 * TRIG IS 1D AND SPLIT ONLY
 * -------------------------
 * A 2D or interleaved trig request is refused LOUDLY, above this tier, in the
 * shared dims>=2 and layout guards. Those two refusals are golden-bit cells
 * (REFUSE.dct2.2d, REFUSE.dct2.interleaved), so the refusal itself is pinned.
 *
 * HOW A TRIG PLAN IS KEYED — the trap this file exists to keep straight
 * --------------------------------------------------------------------
 * 🔴 A trig INNER c2c is keyed by (OWNING TRANSFORM, OUTER N, K) — never as an
 * ordinary c2c of the inner size. Keying it by the inner size would collide
 * with a genuine c2c request for that size and hand one family the other's
 * verdict. The inner size itself derives from vw2_stride_trig_inner_n.
 *
 * The inner c2c cell rides c2c wisdom (calibrate-on-miss at rigor). MT is
 * internal: the inner r2c / c2c threads over K.
 *
 * 🔴 THIS FAMILY HAS NO BANKED WISDOM
 * -----------------------------------
 * The store holds 539 cells and ZERO of them are trig. Every consequence
 * follows from that one fact:
 *
 *   - the fingerprint replay has almost nothing to replay here;
 *   - several trig configs RACE at create, so their output is chosen by the
 *     clock and a single digest of one is a coin flip, not a baseline;
 *   - the migration plan calls step 27 its least-protected step, correctly.
 *
 * The protection actually in place for the move is build_tuned/trig_capture.py
 * — output digests over one process per observation, with raced cells recorded
 * AS raced rather than sampled. That proves the tier still produces what it
 * produced; it does NOT prove the tier correct. The naive O(N^2) reference
 * that would is still absent, deliberately: the plane-role contract is not
 * stated plainly enough in include/vfft.h to encode without guessing, and a
 * wrong expectation baked into a baseline is worse than a missing one.
 *
 * POSITION IN vfft.c IS LOAD-BEARING
 * ----------------------------------
 * Not a standalone header. The builders call _calibrate_c2c, _inner_c2c and
 * _vw2_persist, all file-scope statics defined earlier in vfft.c, so this must
 * be included after those and before _vfft_create_inner. Moving the cluster
 * down from its original position is safe precisely because nothing it calls
 * is defined below it.
 *
 * The six parameters are the create block's complete free-variable set,
 * derived rather than guessed: cfg, ob, W, reg, N, K.
 */
#ifndef VFFT_TRANSFORMS_TRIG_CREATE_H
#define VFFT_TRANSFORMS_TRIG_CREATE_H

/* ---- the builder cluster (whole-function move) ---- */

static int _vw2_t_of_trig(vfft_transform_t t)
{
    switch (t) {
    case VFFT_DCT1: return VW2_T_DCT1;
    case VFFT_DCT2: return VW2_T_DCT2;
    case VFFT_DCT3: return VW2_T_DCT3;
    case VFFT_DCT4: return VW2_T_DCT4;
    case VFFT_DST1: return VW2_T_DST1;
    case VFFT_DST2: return VW2_T_DST2;
    case VFFT_DST3: return VW2_T_DST3;
    case VFFT_DHT:  return VW2_T_DHT;
    default:        return VW2_T_NONE;
    }
}

/* The trig-owned twin of _inner_c2c: same calibrate-on-miss contract, but
 * the verdict is keyed (owning transform, OUTER N, K). */
static stride_plan_t *_inner_c2c_trig(struct vfft_wisdom_s *W,
                                      const vfft_config_t *cfg,
                                      vfft_transform_t owner, int outerN,
                                      int innerN, size_t K, vfft_rigor_t rigor,
                                      const vfft_proto_registry_t *reg,
                                      vfft_proto_wisdom_t *cw, int recalib)
{
    const int wt = _vw2_t_of_trig(owner);
    vfft_proto_wisdom_entry_t ne;
    int have;
    if (wt == VW2_T_NONE || W->vw2_off_stride)
        return _inner_c2c(W, innerN, K, rigor, reg, cw, recalib); /* old path */
    have = !recalib && vw2_stride_lookup_t(&W->vw2, wt, outerN, K, &ne);
    if (have)
        vfft_proto_wisdom_set(cw, &ne);      /* seed auto_plan's process cache */
    else if (_calibrate_c2c(innerN, K, rigor, reg, &ne) == 0)
    {
        vfft_proto_wisdom_add(cw, &ne, 1);
        vw2_stride_bank_entry_t(&W->vw2, &ne, wt, outerN);
        _vw2_persist(W, cfg);
    }
    return vfft_proto_auto_plan(innerN, K, reg, cw);
}

/* JIT the inner c2c of a trig stride-r2c plan: resolve the inner's JIT fwd/bwd and
 * wire them in (the trig forward drives the inner via the r2c forward, so inner_jit_fwd
 * is what runs; bwd set for completeness). NULL-safe / no-op without VFFT_USE_JIT —
 * exactly the Rader/Bluestein inner-JIT pattern. Transparent: no behavior change beyond
 * running the JIT'd inner codelets when JIT is compiled in. */
static inline void _trig_r2c_set_inner_jit(stride_plan_t *r, stride_plan_t *ic)
{
#ifdef VFFT_USE_JIT
    stride_r2c_set_inner_jit_fwd(r, vfft_proto_plan_jit_fwd(ic));
    stride_r2c_set_inner_jit_bwd(r, vfft_proto_plan_jit_bwd(ic));
#else
    (void)r;
    (void)ic;
#endif
}

/* Build the trig stride_plan_t. Owns its inner plans (freed via stride_plan_destroy).
 * The inner real-FFT (r2c) / complex-FFT (DCT-IV) rides the c2c wisdom (calibrate-on-
 * miss) AND, when JIT is compiled in, its inner c2c runs the JIT'd executor. */
static stride_plan_t *_build_trig(struct vfft_wisdom_s *W, const vfft_config_t *cfg,
                                  vfft_transform_t t, int N, size_t K, vfft_rigor_t rigor,
                                  const vfft_proto_registry_t *reg,
                                  vfft_proto_wisdom_t *cw, int recalib)
{
    if (t == VFFT_DCT4)
    { /* inner = half-N complex FFT (driven backward) */
        stride_plan_t *c2c = _inner_c2c_trig(W, cfg, t, N, N / 2, K, rigor, reg, cw, recalib);
        if (!c2c)
            return NULL;
        stride_plan_t *dp = stride_dct4_plan(N, K, c2c);
#ifdef VFFT_USE_JIT
        if (dp) /* DCT-IV drives the inner FFT backward -> JIT its bwd */
            stride_dct4_set_inner_jit_bwd(dp, vfft_proto_plan_jit_bwd(c2c));
#endif
        return dp;
    }
    if (t == VFFT_DCT1 || t == VFFT_DST1)
    { /* boundary r2c of M */
        int M = (t == VFFT_DCT1) ? 2 * (N - 1) : 2 * (N + 1);
        stride_plan_t *ic = _inner_c2c_trig(W, cfg, t, N, M / 2, K, rigor, reg, cw, recalib);
        stride_plan_t *r = ic ? stride_r2c_plan(M, K, K, ic) : NULL;
        if (!r)
            return NULL;
        _trig_r2c_set_inner_jit(r, ic);
        return (t == VFFT_DCT1) ? stride_dct1_plan(N, K, r) : stride_dst1_plan(N, K, r);
    }
    /* DCT-II/III, DST-II/III, DHT — all start from an N-point r2c plan. */
    stride_plan_t *ic = _inner_c2c_trig(W, cfg, t, N, N / 2, K, rigor, reg, cw, recalib);
    stride_plan_t *r = ic ? stride_r2c_plan(N, K, K, ic) : NULL;
    if (!r)
        return NULL;
    _trig_r2c_set_inner_jit(r, ic);
    if (t == VFFT_DHT)
        return stride_dht_plan(N, K, r);
    stride_plan_t *dct2 = stride_dct2_plan(N, K, r);
    if (t == VFFT_DCT2 || t == VFFT_DCT3)
        return dct2;                                   /* DCT-III = dct2 plan, exec dct3 */
    return dct2 ? stride_dst2_plan(N, K, dct2) : NULL; /* DST-II/III wrap DCT-II */
}

/* ---- the create tier (slice) ---- */

static vfft_plan _vfft_create_trig(const vfft_config_t *cfg,
                                   vfft_batch ob,
                                   struct vfft_wisdom_s *W,
                                   const vfft_proto_registry_t *reg,
                                   int N, size_t K)
{
    if (_VFFT_IS_TRIG(cfg->transform))
    {
        /* PADDED (opt-in): build at Kp (aligned) so the trig stride plan strides the caller's
         * Kp-wide real in/out buffers exactly. Pad-only (the trig stride_r2c_plan bakes K, like
         * r2c). BONUS: the odd-K trig TAIL (stride_r2c_plan pre/post) is an unbuilt phase-2 gap,
         * so padding is the ONLY correct full-SIMD trig for misaligned K — it sidesteps the tail
         * by building aligned. Cascade regime (small Kp). */
        size_t bK = K;
        int padded = 0;
        if (ob)
        {
            vfft_batch b = ob;
            if (b->xform != (int)cfg->transform || b->N != N || b->K != K)
            {
                _vfft_warn("vfft_create: config.batch does not match this %s descriptor "
                           "(batch: %s N=%d K=%zu; config: %s N=%d K=%zu) — allocate with "
                           "vfft_alloc_batch_for(THIS config)",
                           _vfft_tname(cfg->transform), _vfft_tname(b->xform), b->N, b->K,
                           _vfft_tname(cfg->transform), N, K);
                return NULL;
            }
            bK = b->Kp;
            padded = 1;
        }
        /* Odd/misaligned tight K now works: the stride r2c inner routes a non-VW-aligned B
         * through its explicit-pack fallback (rem-aware codelet tail + scalar unpack) instead
         * of the crashing fused stage — see _r2c_worker_fwd/_bwd in r2c.h. (Padded builds at
         * VW-aligned Kp regardless.) */
        stride_plan_t *tp = _build_trig(W, cfg, cfg->transform, N, bK, cfg->rigor, reg,
                                        &W->c2c, cfg->recalibrate);
        if (W->path_c2c[0])
            _vw2_persist(W, cfg); /* persist inner cells (guarded, wave-4) */
        if (!tp)
            return NULL;
        struct vfft_plan_s *h = (struct vfft_plan_s *)calloc(1, sizeof *h);
        if (!h)
        {
            stride_plan_destroy(tp);
            return NULL;
        }
        h->transform = cfg->transform;
        h->placement = cfg->placement;
        h->layout = (int)cfg->layout; /* SPLIT by the trig+IL gate up front */
        h->N = N;
        h->K = K;
        h->nthreads = _vfft_plan_threads(cfg);
        h->tplan = tp;
        h->padded = padded;
        h->exec_me = (int)bK;
        return h;
    }
    return NULL; /* unreachable: the one call site guards on the same
                  * condition, and every path in the block above returns. */
}

#endif /* VFFT_TRANSFORMS_TRIG_CREATE_H */
