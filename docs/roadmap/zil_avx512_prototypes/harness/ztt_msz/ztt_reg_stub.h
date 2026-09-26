/* harness stub: zero fused cells (every plan STAGED via force_staged) */
#ifndef ZTT_REG_STUB_H
#define ZTT_REG_STUB_H
#include <stddef.h>
typedef void (*vfft_ztt_fn)(const double *zin, double *zout, double *plane, const double *tw, const size_t *rb, size_t tile);
typedef void (*vfft_zttp_fn)(const double *zin, double *zout, const double *tw, size_t tile);
typedef struct { int n, nf; int chain[7]; vfft_ztt_fn fwd_dest, fwd_plane, bwd_dest, bwd_plane; vfft_zttp_fn fwd_scr, bwd_scr; } vfft_ztt_cell_t;
#define VFFT_ZTT_NCELLS 0
static const vfft_ztt_cell_t vfft_ztt_cells[1];
#endif
