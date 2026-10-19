/* vfft_api.h -- THE PUBLIC API, version 2 (include/vfft.h; docs/roadmap/api_v2_design.md, 2026-10-15).
 *
 * The eight functions over the library's own create, execute and plan facts. The public
 * configuration is translated here, once, into the request the engines read (vfft_cfg_t,
 * common/plan/vfft_cfg.h); nothing below this file sees a vfft_config_t. vfft_destroy,
 * vfft_malloc and vfft_free are defined where they always were (vfft_execute.h,
 * vfft_memory.h); this file adds the rest.
 *
 * THE TRANSLATION, the owner's rulings of 2026-10-15:
 *   - the zeroed public config is INTERLEAVED, OUT OF PLACE, NATURAL, one thread;
 *   - NATURAL is the default order at every rank and layout: the request is the library's
 *     DEFAULT wherever DEFAULT is natural, and an explicit NATURAL where the layout's own order
 *     is the scrambled comb (SPLIT C2C at rank 2 and above; at rank 3 and 4 SPLIT serves the
 *     scrambled comb only, so such a request is refused and says to ask SCRAMBLED);
 *   - real-to-real transforms take no layout, order or batch geometry: SPLIT internally, DEFAULT;
 *   - the batch values are the library's own (DEFAULT: contiguous for INTERLEAVED C2C, lane-major
 *     for real and for SPLIT);
 *   - nthreads 0 or 1 is one thread (no process-wide setting);
 *   - every create plans PATIENT (ruled 2026-10-17): VFFT_FLAG_PATIENT names that default and
 *     VFFT_FLAG_EXHAUSTIVE is refused;
 *   - the other flags become recalibrate, destroy_input and owned_buffers;
 *   - wisdom_dir becomes the store for that directory, loaded once per process and kept
 *     (a create saves its winners into it before returning).
 *
 * THE REFUSAL RECORD: _vfft_warn (common/support/diag.h) writes every message into a per-thread
 * buffer and prints it only under VFFT_FLAG_VERBOSE (this create) or VFFT_VERBOSE (the process);
 * vfft_error() returns the buffer. A create starts it empty and leaves it empty when it succeeds.
 *
 * Included at the end of vfft.c: after the front door, the execute door and the plan facts. */
#ifndef VFFT_API_H
#define VFFT_API_H

/* ── the wisdom store of a directory: loaded once per process, kept for its life ── */
#define VFFT_API_MAX_STORES 64
static struct
{
    char path[512];
    vfft_wisdom *w;
} _vfft_api_stores[VFFT_API_MAX_STORES];
static int _vfft_api_nstores = 0;
/* under the create lock */
static vfft_wisdom *_vfft_store_for_dir(const char *dir)
{
    int i;
    vfft_wisdom *w;
    for (i = 0; i < _vfft_api_nstores; i++)
        if (!strcmp(_vfft_api_stores[i].path, dir))
            return _vfft_api_stores[i].w;
    if (strlen(dir) >= sizeof _vfft_api_stores[0].path)
    {
        _vfft_warn("vfft_create: wisdom_dir is %zu characters; at most %zu are served",
                   strlen(dir), sizeof _vfft_api_stores[0].path - 1);
        return NULL;
    }
    if (_vfft_api_nstores == VFFT_API_MAX_STORES)
    {
        _vfft_warn("vfft_create: %d wisdom directories are already open in this process; "
                   "a plan uses one of them or the library's own store",
                   VFFT_API_MAX_STORES);
        return NULL;
    }
    w = _vfft_wisdom_load(dir);
    if (!w)
    {
        _vfft_warn("vfft_create: out of memory opening the wisdom store %s", dir);
        return NULL;
    }
    snprintf(_vfft_api_stores[_vfft_api_nstores].path, sizeof _vfft_api_stores[0].path, "%s", dir);
    _vfft_api_stores[_vfft_api_nstores].w = w;
    _vfft_api_nstores++;
    return w;
}

/* the diagnostics header's re-read: the kept store of `dir` closed, so the next create
 * naming `dir` loads the files (a plan never holds the caller's store: each keeps its own
 * copies, so closing it under live plans is safe) */
int vfft_wisdom_reread(const char *dir)
{
    int i;
    if (!dir)
        return -1;
    _vfft_create_lock();
    for (i = 0; i < _vfft_api_nstores; i++)
        if (!strcmp(_vfft_api_stores[i].path, dir))
        {
            _vfft_wisdom_free(_vfft_api_stores[i].w);
            _vfft_api_stores[i] = _vfft_api_stores[--_vfft_api_nstores];
            break;
        }
    _vfft_create_unlock();
    return 0;
}

/* ── the public configuration into the request ── */
static int _vfft_api_translate(const vfft_config_t *p, vfft_cfg_t *c)
{
    const unsigned known = VFFT_FLAG_PATIENT | VFFT_FLAG_EXHAUSTIVE | VFFT_FLAG_RECALIBRATE |
                           VFFT_FLAG_DESTROY_INPUT | VFFT_FLAG_OWN_PITCH | VFFT_FLAG_VERBOSE;
    int trig, rank;
    memset(c, 0, sizeof *c);
    if ((unsigned)p->transform > (unsigned)VFFT_DHT)
    {
        _vfft_warn("vfft_create: invalid transform %d (valid: VFFT_C2C .. VFFT_DHT)", (int)p->transform);
        return 0;
    }
    if (p->layout != VFFT_LAYOUT_INTERLEAVED && p->layout != VFFT_LAYOUT_SPLIT)
    {
        _vfft_warn("vfft_create: invalid layout %d (valid: VFFT_LAYOUT_INTERLEAVED, VFFT_LAYOUT_SPLIT)", (int)p->layout);
        return 0;
    }
    if (p->placement != VFFT_PLACEMENT_OUT && p->placement != VFFT_PLACEMENT_IN)
    {
        _vfft_warn("vfft_create: invalid placement %d (valid: VFFT_PLACEMENT_OUT, VFFT_PLACEMENT_IN)", (int)p->placement);
        return 0;
    }
    if (p->order != VFFT_ORDER_NATURAL && p->order != VFFT_ORDER_SCRAMBLED)
    {
        _vfft_warn("vfft_create: invalid order %d (valid: VFFT_ORDER_NATURAL, VFFT_ORDER_SCRAMBLED)", (int)p->order);
        return 0;
    }
    if (p->batch != VFFT_BATCH_DEFAULT && p->batch != VFFT_BATCH_CONTIGUOUS && p->batch != VFFT_BATCH_LANE_MAJOR)
    {
        _vfft_warn("vfft_create: invalid batch %d (valid: VFFT_BATCH_DEFAULT, VFFT_BATCH_CONTIGUOUS, "
                   "VFFT_BATCH_LANE_MAJOR)", (int)p->batch);
        return 0;
    }
    if (p->flags & ~known)
    {
        _vfft_warn("vfft_create: unknown flag bits 0x%x", p->flags & ~known);
        return 0;
    }
    if (p->flags & VFFT_FLAG_EXHAUSTIVE)
    {
        _vfft_warn("vfft_create: VFFT_FLAG_EXHAUSTIVE is not served (every create plans PATIENT; drop the flag)");
        return 0;
    }
    if (p->rank < 0 || p->rank > 4)
    {
        _vfft_warn("vfft_create: rank %d out of range (0 or 1 is 1D; 2, 3, 4)", p->rank);
        return 0;
    }
    if (p->nthreads < 0)
    {
        _vfft_warn("vfft_create: nthreads %d is negative (0 or 1 is one thread)", p->nthreads);
        return 0;
    }
    trig = _VFFT_IS_TRIG(p->transform);
    rank = p->rank < 1 ? 1 : p->rank;
    c->transform = p->transform;
    c->dims = rank;
    memcpy(c->n, p->n, sizeof c->n);
    c->howmany = p->howmany ? p->howmany : 1;
    c->layout = (trig || p->layout == VFFT_LAYOUT_SPLIT) ? VFI_LAYOUT_SPLIT : VFI_LAYOUT_IL;
    c->placement = p->placement == VFFT_PLACEMENT_IN ? VFFT_INPLACE : VFFT_OUTOFPLACE;
    if (trig)
        c->order = VFI_ORDER_DEFAULT;          /* no order axis: natural bins */
    else if (p->order == VFFT_ORDER_SCRAMBLED)
        c->order = VFI_ORDER_SCRAMBLED;
    else if (c->layout == VFI_LAYOUT_SPLIT && rank >= 2 && p->transform == VFFT_C2C)
        c->order = VFI_ORDER_NATURAL;          /* the split tiers' own order is the scrambled comb */
    else
        c->order = VFI_ORDER_DEFAULT;          /* natural is the layout's own */
    c->batch_geom = trig ? VFFT_BATCH_DEFAULT : (int)p->batch;
    c->nthreads = p->nthreads > 1 ? p->nthreads : 1;
    c->rigor = VFFT_PATIENT;               /* every create, VFFT_FLAG_PATIENT or not */
    c->recalibrate = (p->flags & VFFT_FLAG_RECALIBRATE) != 0;
    c->destroy_input = (p->flags & VFFT_FLAG_DESTROY_INPUT) != 0;
    c->owned_buffers = (p->flags & VFFT_FLAG_OWN_PITCH) != 0;
    if (p->wisdom_dir)
    {
        c->wisdom = _vfft_store_for_dir(p->wisdom_dir);
        if (!c->wisdom)
            return 0;
    }
    return 1;
}

vfft_plan vfft_create(const vfft_config_t *config)
{
    vfft_cfg_t c;
    vfft_plan h = NULL;
    const int v0 = _vfft_verbose_tls;
    _vfft_err_buf[0] = '\0';
    if (!config)
    {
        _vfft_warn("vfft_create: NULL config");
        return NULL;
    }
    _vfft_verbose_tls = (config->flags & VFFT_FLAG_VERBOSE) != 0;
    _vfft_create_lock();   /* the store table is the lock's too (recursive: the create takes it again) */
    if (_vfft_api_translate(config, &c))
    {
        h = _vfft_create_cfg(&c);
        if (h && h->nthreads > 1)   /* no lean entry: a threaded plan's execute pins the calling thread
                                     * around the door and puts it back (vfft_execute.h, _vfft_exec6) */
            h->k1_go[VFFT_FORWARD] = h->k1_go[VFFT_BACKWARD] = NULL;
    }
    _vfft_create_unlock();
    if (h)
        _vfft_err_buf[0] = '\0';
    _vfft_verbose_tls = v0;
    return h;
}

/* ── execute: the lean door for INTERLEAVED and real-to-real plans; the split door ── */
static __attribute__((noinline)) void _vfft_api_execute_slow(vfft_plan h, vfft_dir_t dir, double *in, double *out)
{
    if (h && h->layout == VFI_LAYOUT_SPLIT && !_VFFT_IS_TRIG(h->transform))
    {
        _vfft_warn("vfft_execute: a SPLIT plan runs through vfft_execute_split(); nothing executed");
        return;
    }
    _vfft_exec6(h, dir, in, NULL, out, NULL);
}
/* THE BOUND FAST PATH (the lean door, 2026-10-11): a plan with a never-declining executor per
 * direction is entered by a jump after four compares; everything else takes the full door. A
 * threaded plan carries no entry (vfft_create): it takes the full door, where the calling thread
 * is pinned to the pool's reserved core for the execute (vfft_execute.h, _vfft_exec6). */
void vfft_execute(vfft_plan h, vfft_dir_t dir, const void *in, void *out)
{
    double *s = (double *)in, *d = (double *)out;
    if (h && (unsigned)dir <= (unsigned)VFFT_BACKWARD && s && d &&
        (h->placement == VFFT_INPLACE ? d == s : (d != s || h->transform == VFFT_C2C)) && h->k1_go[dir])
    {
        h->k1_go[dir](h, dir, s, d);
        return;
    }
    _vfft_api_execute_slow(h, dir, s, d);
}
void vfft_execute_split(vfft_plan h, vfft_dir_t dir,
                        const double *in_re, const double *in_im,
                        double *out_re, double *out_im)
{
    if (h && (h->layout != VFI_LAYOUT_SPLIT || _VFFT_IS_TRIG(h->transform)))
    {
        _vfft_warn("vfft_execute_split: %s plan runs through vfft_execute(); nothing executed",
                   _VFFT_IS_TRIG(h->transform) ? "a real-to-real" : "an INTERLEAVED");
        return;
    }
    _vfft_exec6(h, dir, (double *)in_re, (double *)in_im, out_re, out_im);
}

/* ── the plan's facts ── */
/* the doubles execute reads and writes (per plane under SPLIT); 0 when the product overflows */
static void _vfft_api_sizes(const struct vfft_plan_s *h, size_t *in, size_t *out)
{
    const int real = (h->transform == VFFT_R2C || h->transform == VFFT_C2R);
    const int trig = _VFFT_IS_TRIG(h->transform);
    const int il = (h->layout == VFI_LAYOUT_IL);
    const size_t K = h->K > 0 ? h->K : 1;
    const size_t Kp = (!il && h->own_batch) ? h->own_batch->Kp : K;
    size_t lead = (size_t)h->N, last = (size_t)h->N, pts, half, a, b;
    /* the extents: the last axis and the product of the ones before it */
    if (h->N4 > 0)      { lead = (size_t)h->N * (size_t)h->N2 * (size_t)h->N3; last = (size_t)h->N4; }
    else if (h->N3 > 0) { lead = (size_t)h->N * (size_t)h->N2; last = (size_t)h->N3; }
    else if (h->N2 > 0) { lead = (size_t)h->N; last = (size_t)h->N2; }
    else                { lead = 1; last = (size_t)h->N; }
    pts = lead * last;
    half = lead * (last / 2 + 1);
    *in = *out = 0;
    if (trig)
    {   /* N reals in and out (SPLIT internally: one plane each) */
        a = b = pts * Kp;
    }
    else if (!real)
    {   /* complex points: two doubles each interleaved, one per plane split */
        a = b = pts * (il ? K * 2 : Kp);
    }
    else
    {   /* real points against the half spectrum; in place interleaved the plan's pitch (door 2) */
        size_t hrow = last / 2 + 1;
        if (il && h->N2 > 0 && h->N3 == 0 && h->il2d_ipP)
            hrow = h->il2d_ipP;
        if (il && h->N3 > 0 && h->ilndr)
            hrow = (size_t)h->ilndr->hp3;
        half = lead * hrow;
        a = pts * (il ? K : Kp);
        b = half * (il ? K * 2 : Kp);
        if (h->transform == VFFT_C2R)
        {
            const size_t t = a;
            a = b;
            b = t;
        }
    }
    if (h->placement == VFFT_INPLACE && il && real)
        a = b = a > b ? a : b;   /* one buffer serves both roles: the larger, the real in-place plane */
    *in = a;
    *out = b;
}
/* the stride the plan walks: a batch's element stride (complex elements interleaved, doubles in a
 * split plane), or an in-place rank-2/3 real plan's row pitch in doubles */
static size_t _vfft_api_stride(const struct vfft_plan_s *h)
{
    const int real = (h->transform == VFFT_R2C || h->transform == VFFT_C2R);
    if (h->own_pitch)
        return h->own_pitch;   /* door 2: the pitch the plan chose (VFFT_FLAG_OWN_PITCH) */
    if (h->lm)
        return h->K > 0 ? h->K : 1;   /* a lane-major batch at any rank: the lane stride */
    if (real && h->layout == VFI_LAYOUT_IL && h->placement == VFFT_INPLACE && h->N2 > 0)
    {
        if (h->N3 > 0)
            return 2 * (h->ilndr ? (size_t)h->ilndr->hp3 : (size_t)(h->N3 / 2 + 1));
        return 2 * (h->il2d_ipP ? (size_t)h->il2d_ipP : (size_t)(h->N2 / 2 + 1));
    }
    if (h->own_batch)
        return h->own_batch->Kp;   /* the split padded batch: the stride its pad race chose */
    return h->K > 0 ? h->K : 1;
}
int vfft_info(vfft_plan p, vfft_info_t *out)
{
    const struct vfft_plan_s *h = (const struct vfft_plan_s *)p;
    if (!out)
        return -1;
    memset(out, 0, sizeof *out);
    out->version = VFFT_VERSION_STRING;
    out->alignment = (size_t)VFFT_ALIGNMENT;
    out->cores = vfft_topology()->n_p > 0 ? vfft_topology()->n_p : 1;   /* cpu_topology.h: the P-cores, every core when not hybrid */
    out->route = "-";
    if (!h)
        return 0;
    out->route = _vfft_plan_route(p);
    out->nthreads = h->nthreads > 1 ? h->nthreads : 1;
    out->stride = _vfft_api_stride(h);
    _vfft_api_sizes(h, &out->in_doubles, &out->out_doubles);
    return 0;
}
const char *vfft_error(void)
{
    return _vfft_err_buf;
}

#endif /* VFFT_API_H */
