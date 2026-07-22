/* adopt_wisdom.h — §6a49/Q3: persisted strided-adoption decisions.
 *
 * Every 2D/ND r2c create runs measured A/B gates (§6a39/41/47) costing
 * tens of ms. This sidecar persists the per-shape verdicts so warm creates
 * skip the arms entirely. Deliberately NOT part of the binary wisdom
 * bundles: a human-readable, versioned, line-oriented text file that is
 * trivially inspectable and safely ignorable.
 *
 * Keying: (kind, rows, NL, blk). blk is the edition block quantum (4 =
 * avx2, 8 = avx512 build) — decisions are edition-specific and builds get
 * separate records. Values: fwd/bwd adoption booleans.
 *
 * Path: $VFFT_ADOPT_WISDOM_DIR/ (its OWN env — deliberately NOT VFFT_WISDOM_DIR, which activates the heavy bundle-calibration machinery) strided_adopt.wis. Env unset => fully disabled
 * (lookup misses, record no-ops) — the default matches the engine's other
 * wisdom paths. Corrupt lines are skipped; writes go through tmp+rename.
 * Single-threaded create assumption (as the rest of plan creation).
 */
#ifndef VFFT_ADOPT_WISDOM_H
#define VFFT_ADOPT_WISDOM_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define _VAW_MAX 512

typedef struct {
    char kind[4];
    int a, b, blk;
    int fwd, bwd;
} _vaw_rec_t;

static _vaw_rec_t _vaw_tab[_VAW_MAX];
static int _vaw_n = 0;
static int _vaw_loaded = 0;

static inline const char *_vaw_path(char *buf, size_t cap)
{
    const char *d = getenv("VFFT_ADOPT_WISDOM_DIR");
    if (!d || !d[0]) return NULL;
    snprintf(buf, cap, "%s/strided_adopt.wis", d);
    return buf;
}

static inline void _vaw_load(void)
{
    if (_vaw_loaded) return;
    _vaw_loaded = 1;
    char pb[512];
    const char *p = _vaw_path(pb, sizeof pb);
    if (!p) return;
    FILE *f = fopen(p, "r");
    if (!f) return;
    char line[128];
    if (!fgets(line, sizeof line, f) || strncmp(line, "vfftaw1", 7)) {
        fclose(f);
        return;                    /* wrong version: ignore whole file */
    }
    while (fgets(line, sizeof line, f) && _vaw_n < _VAW_MAX) {
        _vaw_rec_t r;
        if (sscanf(line, "%3s %d %d %d %d %d",
                   r.kind, &r.a, &r.b, &r.blk, &r.fwd, &r.bwd) == 6)
            _vaw_tab[_vaw_n++] = r;
    }
    fclose(f);
}

static inline int vfft_adopt_lookup(const char *kind, int a, int b, int blk,
                                    int *fwd, int *bwd)
{
    _vaw_load();
    for (int i = 0; i < _vaw_n; i++)
        if (!strcmp(_vaw_tab[i].kind, kind) && _vaw_tab[i].a == a
            && _vaw_tab[i].b == b && _vaw_tab[i].blk == blk) {
            *fwd = _vaw_tab[i].fwd;
            *bwd = _vaw_tab[i].bwd;
            return 1;
        }
    return 0;
}

static inline void vfft_adopt_record(const char *kind, int a, int b, int blk,
                                     int fwd, int bwd)
{
    _vaw_load();
    char pb[512];
    const char *p = _vaw_path(pb, sizeof pb);
    if (!p) return;
    for (int i = 0; i < _vaw_n; i++)
        if (!strcmp(_vaw_tab[i].kind, kind) && _vaw_tab[i].a == a
            && _vaw_tab[i].b == b && _vaw_tab[i].blk == blk) {
            _vaw_tab[i].fwd = fwd;
            _vaw_tab[i].bwd = bwd;
            goto write;
        }
    if (_vaw_n < _VAW_MAX) {
        _vaw_rec_t r;
        snprintf(r.kind, sizeof r.kind, "%s", kind);
        r.a = a; r.b = b; r.blk = blk; r.fwd = fwd; r.bwd = bwd;
        _vaw_tab[_vaw_n++] = r;
    }
write:;
    char tp[520];
    snprintf(tp, sizeof tp, "%s.tmp", p);
    FILE *f = fopen(tp, "w");
    if (!f) return;
    fputs("vfftaw1\n", f);
    for (int i = 0; i < _vaw_n; i++)
        fprintf(f, "%s %d %d %d %d %d\n", _vaw_tab[i].kind, _vaw_tab[i].a,
                _vaw_tab[i].b, _vaw_tab[i].blk, _vaw_tab[i].fwd,
                _vaw_tab[i].bwd);
    fclose(f);
    rename(tp, p);
}

#endif /* VFFT_ADOPT_WISDOM_H */
