/* nat_modes.h - the natural-order mode ids stored in @nat wisdom rows.
 *
 * Carved verbatim out of split/planning/wisdom_reader.h (layout separation
 * phase 4): the split modes (FREE/SCR/PURE_CYCLE/PSWAP) and three
 * interleaved ones (ZCASC/ILP/CONV) share this one numbering today. Decision
 * D2 (owner, 2026-09-27) gives the interleaved in-place door its own lay=il
 * row; the IL values then leave this list. Numbers are a file format. */
#ifndef VFFT_NAT_MODES_H
#define VFFT_NAT_MODES_H

/* Natural-order modes. A reader that meets an unknown mode re-measures:
 * degraded, never wrong. 2 (LEAF_IP) is retired but NEVER reused — old
 * files may still carry it with the old meaning. */
enum { VFFT_NAT_UNSET = 0, VFFT_NAT_FREE = 1, VFFT_NAT_LEAF_IP = 2,
       VFFT_NAT_SCR = 3, VFFT_NAT_PURE_CYCLE = 4, VFFT_NAT_PSWAP = 5,
       /* ZCASC: the verdict of the K=1 interleaved zturn cascade with the
        * natural terminator (no reorder pass). The cascade engine is retired;
        * the value stays so stored records keep parsing. The @nat entry
        * stored only the verdict; the chain came from the kind-4 oop line. */
       VFFT_NAT_ZCASC = 6,
       /* ILP: the K=1 interleaved IN-PLACE cells served by the native IL
        * engines (mono structurally refuses aliasing) — natural output, no
        * tape, no layout conversion. An explicit-SCRAMBLED in-place create
        * attaches only on a hit (identity permutation), which keeps @nat
        * single-writer. */
       VFFT_NAT_ILP = 7,
       /* CONV: the banked LOSS of the scrambled in-place IL race — "raced,
        * the convert incumbent won" — in the ord=scr mode cell only (the @nat
        * natural cells never carry it), so a lost race is not re-run on every
        * create. */
       VFFT_NAT_CONV = 8 };

#endif /* VFFT_NAT_MODES_H */
