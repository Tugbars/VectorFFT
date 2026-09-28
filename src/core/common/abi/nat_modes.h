/* nat_modes.h - the natural-order mode ids stored in split @nat wisdom rows.
 *
 * Carved verbatim out of split/planning/wisdom_reader.h (layout separation
 * phase 4). Owner decision D2 (2026-09-27): the interleaved values ZCASC (6),
 * ILP (7) and CONV (8) left this list; the IL in-place door keeps its verdict
 * on its own kind-3 lay=il row (il/rank1/c2c_ip_create_il.h). Numbers are a
 * file format: 2, 6, 7 and 8 are retired and NEVER reused. A row carrying 6,
 * 7 or 8 is no split verdict: the wisdom2 reader reads it as absent, and the
 * split in-place natural arm ignores it in the frozen spike table (which still
 * carries such rows and is left as it is). Read as a split verdict it made
 * that arm skip its reorder tape, and a NATURAL request came back in
 * scrambled order (the shipped rows N = 128/256/255/512/1024, K = 1). */
#ifndef VFFT_NAT_MODES_H
#define VFFT_NAT_MODES_H

/* Natural-order modes. A reader that meets an unknown mode re-measures:
 * degraded, never wrong. 2 (LEAF_IP) is retired but NEVER reused — old
 * files may still carry it with the old meaning. */
enum { VFFT_NAT_UNSET = 0, VFFT_NAT_FREE = 1, VFFT_NAT_LEAF_IP = 2,
       VFFT_NAT_SCR = 3, VFFT_NAT_PURE_CYCLE = 4, VFFT_NAT_PSWAP = 5,
       VFFT_NAT_MAX = VFFT_NAT_PSWAP /* the last mode; 6-8 retired by D2 */ };

#endif /* VFFT_NAT_MODES_H */
