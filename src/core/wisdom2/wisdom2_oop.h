/* wisdom2_oop.h — the front door's OOP-family wisdom include.
 *
 * Layout separation phase 5 (the kind-3 record split, owner's option (a),
 * 2026-09-27). What was here is now three headers:
 *   common/wisdom/wisdom2_oop_legacy.h  the FROZEN oop_wisdom.txt format: the
 *                                       dual entry, table, loader, lookups
 *   split/wisdom/wisdom2_oop_split.h    the split record + codec
 *   il/wisdom/wisdom2_oop_il.h          the interleaved record + codec
 * The front door (vfft.c, vfft_internal.h, the create tiers) includes this one
 * header; a layout's own code includes its own codec only. */
#ifndef VFFT_OOP_WISDOM_H
#define VFFT_OOP_WISDOM_H

#include "wisdom2_oop_legacy.h"
#include "wisdom2_oop_split.h"
#include "wisdom2_oop_il.h"

#endif /* VFFT_OOP_WISDOM_H */
