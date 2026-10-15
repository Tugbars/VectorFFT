/* vfft_cfg.h -- THE REQUEST AS THE ENGINES READ IT (2026-10-15, the public API's version 2).
 *
 * The public configuration (include/vfft.h, vfft_config_t) is translated once, at the front
 * door (vfft_api.h), into this struct, and every tier, engine and child create below the door
 * reads only this one. Its constants are the library's own vocabulary and keep the meanings the
 * engines were written against: a zeroed vfft_cfg_t is SPLIT, IN PLACE, the layout's own order,
 * the pool's thread setting -- the opposite of a zeroed public config, which is why the two
 * never share a struct. A child create (a tier's row plan, a composite's inner) builds one of
 * these directly and calls _vfft_create_cfg(); nothing below the door calls vfft_create().
 *
 * Safe to include from a tool that tests the library's internal laws (the policy gate): types
 * and constants only, no function. */
#ifndef VFFT_CFG_H
#define VFFT_CFG_H

#include <stddef.h>
#include "vfft.h"   /* vfft_transform_t, vfft_dir_t, VFFT_BATCH_* (the batch values are shared) */

typedef struct vfft_wisdom_s vfft_wisdom;   /* a wisdom store: the library's, a directory's, a child's */

/* placement */
enum
{
    VFFT_INPLACE = 0,
    VFFT_OUTOFPLACE = 1
};
/* how thoroughly a miss is measured */
typedef enum
{
    VFFT_MEASURE = 0,
    VFFT_PATIENT,
    VFFT_EXHAUSTIVE
} vfft_rigor_t;
/* the complex layout */
enum
{
    VFI_LAYOUT_SPLIT = 0,
    VFI_LAYOUT_IL = 1
};
/* the output order: DEFAULT is the layout's own (natural for INTERLEAVED at every rank and for
 * SPLIT at 1D, the scrambled comb for SPLIT at rank 2 and above) */
enum
{
    VFI_ORDER_DEFAULT = 0,
    VFI_ORDER_NATURAL = 1,
    VFI_ORDER_SCRAMBLED = 2
};

typedef struct
{
    vfft_transform_t transform;
    int placement;     /* VFFT_INPLACE / VFFT_OUTOFPLACE */
    int rigor;         /* vfft_rigor_t */
    int dims;          /* 1..4; 0 is 1D */
    int n[4];
    size_t howmany;
    int owned_buffers; /* the plan chooses its pitch or lane stride (VFFT_FLAG_OWN_PITCH) */
    int nthreads;      /* 0 = the pool's setting */
    int order;         /* VFI_ORDER_* */
    int layout;        /* VFI_LAYOUT_* */
    int batch_geom;    /* VFFT_BATCH_* */
    vfft_wisdom *wisdom; /* NULL = the library's store */
    int recalibrate;
    int wisdom_write;  /* set by the create: whether a winner is saved */
    int destroy_input;
} vfft_cfg_t;

#endif /* VFFT_CFG_H */
