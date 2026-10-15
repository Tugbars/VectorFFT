# The public API, version 2: few functions, fixed defaults

**Status (2026-10-15): built.** The eight functions in `include/vfft.h`, the translation in
`src/core/vfft_api.h`, every tracked caller migrated (`src/tools/baseline` excepted). Owner: "I want to have a very simple API with
few functions, if possible; we do not have to do things their way." The owner's rulings of
the same day: the SPLIT layout gets its own execute call; a zeroed config means
interleaved, out of place, natural; the default order is natural at every rank and layout
("natural is more common in use; scrambled can be faster but it has fewer uses, mostly DSP
people"). The batch defaults stand as ruled the day before: a C2C batch is
transform-contiguous unless it names lane-major, a real batch is lane-major unless it names
transform-contiguous. Strides are not offered (2026-10-14).

The model stays what it is: a configuration names the data contract, create commits it to
one measured plan, execute runs it purely, destroy frees it. What changes is the surface
around that model: every process-wide setting the per-plan configuration already covers is
removed, every plan fact comes back through one query, the planner's policy is one flags
word, and every default has one fixed meaning but the batch geometry's, which follows the
transform as ruled.

The batch field says where the K transforms of a batch sit, and matters only when
`howmany > 1` (at K = 1 the two geometries are the same addressing):

| value | where element e of transform t sits |
|---|---|
| `VFFT_BATCH_CONTIGUOUS` | `z[2 (t N + e)]`: each transform a block of its own, one after another |
| `VFFT_BATCH_LANE_MAJOR` | `z[2 (e K + t)]`: the transforms side by side, sample by sample |
| `VFFT_BATCH_DEFAULT` (0) | contiguous for C2C, lane-major for R2C and C2R |

A real batch's spectrum follows the same rule with N/2 + 1 bins in place of N. At rank 2
and above a batch is contiguous planes; lane-major there is refused until it is built.

## 1. The functions

```c
vfft_plan   vfft_create(const vfft_config_t *cfg);
void        vfft_execute(vfft_plan p, vfft_dir_t dir, const void *in, void *out);
void        vfft_execute_split(vfft_plan p, vfft_dir_t dir,
                               const double *in_re, const double *in_im,
                               double *out_re, double *out_im);
void        vfft_destroy(vfft_plan p);
void       *vfft_malloc(size_t bytes);
void        vfft_free(void *ptr);
int         vfft_info(vfft_plan p, vfft_info_t *out);
const char *vfft_error(void);
```

- `vfft_create` returns the plan, or NULL when the cell is refused; the reason is read with
  `vfft_error`, and printed only under `VFFT_FLAG_VERBOSE`.
- `vfft_execute` serves every INTERLEAVED plan and every real-to-real plan: `in` and `out`
  are the data, complex pairs or reals as the transform says; in place, `out == in`. A
  SPLIT plan given to it is refused, and the reverse.
- `vfft_execute_split` serves every SPLIT plan: the four planes in their roles, unused roles
  NULL (a real input has no imaginary plane, a real output none either).
- An R2C plan runs `VFFT_FORWARD` only and a C2R plan `VFFT_BACKWARD` only; the other
  direction is refused, as today.
- `vfft_info` fills the plan's facts; with `p == NULL` it fills the library's facts only.
- `vfft_error` is the last refusal on the calling thread, static storage, "" when none.

## 2. The configuration

```c
typedef struct {
    vfft_transform_t transform;   /* VFFT_C2C (0), VFFT_R2C, VFFT_C2R, the real-to-real kinds   */
    int        rank;              /* 0 or 1 = 1D; 2, 3, 4                                       */
    int        n[4];              /* the extents, the last axis contiguous                      */
    size_t     howmany;           /* 0 or 1 = one transform                                     */
    vfft_layout_t    layout;      /* VFFT_LAYOUT_INTERLEAVED (0), VFFT_LAYOUT_SPLIT              */
    vfft_placement_t placement;   /* VFFT_PLACEMENT_OUT (0), VFFT_PLACEMENT_IN                  */
    vfft_order_t     order;       /* VFFT_ORDER_NATURAL (0), VFFT_ORDER_SCRAMBLED               */
    vfft_batch_t     batch;       /* VFFT_BATCH_DEFAULT (0): contiguous for C2C, lane-major for  */
                                  /* real; VFFT_BATCH_CONTIGUOUS, VFFT_BATCH_LANE_MAJOR          */
    int        nthreads;          /* 0 or 1 = one thread                                        */
    unsigned   flags;             /* VFFT_FLAG_*                                                 */
    const char *wisdom_dir;       /* NULL = the library's store                                 */
} vfft_config_t;
```

A zeroed configuration with `n[0]` set is one 1D C2C transform, interleaved, out of place,
natural order, one thread, served from the library's store. `howmany = 0` and `rank = 0`
mean one, so nothing has to be written out that a zero cannot say.

The flags:

| flag | replaces | meaning |
|---|---|---|
| `VFFT_FLAG_PATIENT` | `rigor = VFFT_PATIENT` | the wide sweep on a miss |
| `VFFT_FLAG_EXHAUSTIVE` | `rigor = VFFT_EXHAUSTIVE` | the full sweep where a planner has one |
| `VFFT_FLAG_RECALIBRATE` | `recalibrate = 1` | race this cell even on a hit |
| `VFFT_FLAG_DESTROY_INPUT` | `destroy_input = 1` | the plan may overwrite an out-of-place input |
| `VFFT_FLAG_OWN_PITCH` | `owned_buffers = 1` | the plan may choose its row pitch or lane stride; the caller reads it from `vfft_info` and lays the data at it |
| `VFFT_FLAG_VERBOSE` | the printed refusal | print the refusal as well as recording it |

The enums are typed and prefixed: the placement values become `VFFT_PLACEMENT_OUT` and
`VFFT_PLACEMENT_IN`, the rigor enum disappears into the flags (which also ends the
`VFFT_MEASURE` / `VFFT_MEASURE_PIN_*` collision), and `order` and `batch` become typed
fields instead of ints.

## 3. The plan's facts

```c
typedef struct {
    const char *version;        /* "MAJOR.MINOR.PATCH"                                   */
    size_t      alignment;      /* bytes: vfft_malloc's, the buffers the races measured  */
    const char *route;          /* the plan's route name, "-" when it has none           */
    int         nthreads;       /* the count the plan serves at                          */
    size_t      in_doubles;     /* what execute reads, per plane under SPLIT             */
    size_t      out_doubles;    /* what execute writes, per plane under SPLIT            */
    size_t      stride;         /* doubles between rows (rank >= 2 in place) or lane     */
                                /* stride (a batch) the plan walks; the tight value      */
                                /* unless VFFT_FLAG_OWN_PITCH let the plan choose        */
} vfft_info_t;
```

The caller allocates with `vfft_malloc(info.in_doubles * sizeof(double))` and the like; no
buffer is owned by a plan and no second free function exists.

## 4. What goes away

| today | becomes |
|---|---|
| `vfft_wisdom_load`, `vfft_wisdom_save`, `vfft_wisdom_free`, `config.wisdom` | `config.wisdom_dir`; create already saves every winner before it returns; a tool that must read a directory again calls the diagnostics header's `vfft_wisdom_reread` |
| `vfft_set_num_threads`, `vfft_get_num_threads` | `config.nthreads` alone; the diagnostics header's `vfft_pool_threads` reports the pool's size to a tool |
| `vfft_set_thread_block_ms`, `vfft_get_thread_block_ms` | `VFFT_THREAD_BLOCK_MS` |
| `vfft_measure_configure` and its config | the diagnostics header (a tool's surface) |
| `vfft_plan_planes`, `vfft_plan_stride`, `vfft_plan_alloc`, `vfft_buffers_free`, `config.owned_buffers` | `vfft_info` and `VFFT_FLAG_OWN_PITCH` |
| `vfft_plan_route`, `vfft_version`, `vfft_alignment` | fields of `vfft_info` |
| `config.rigor`, `config.recalibrate`, `config.destroy_input` | flags |
| `config.wisdom_write` (retired) | gone |
| `config.dims` | `config.rank` |
| the printed refusal | `vfft_error`, printed under `VFFT_FLAG_VERBOSE` |

The public surface goes from twenty functions to eight. `vfft_diagnostics.h` stays the
tools' header (the engagement counters, the measurement scope as a timing tool, the store
report, the ISA); it is not part of what a user learns.

## 5. The breaks, for the release notes

- The zeroed configuration changes meaning: it was SPLIT, in place; it is INTERLEAVED, out
  of place. The enum values of layout and placement change with it.
- The default order is NATURAL everywhere; SPLIT at rank 2 and above was SCRAMBLED by
  default and is now asked for explicitly.
- A SPLIT plan runs through `vfft_execute_split`; `vfft_execute` takes two pointers.
- The process-wide thread setting is gone: a plan that does not set `nthreads` runs on one
  thread.
- Refusals are no longer printed unless asked.

## 6. The build

1. The header and the front door: the new configuration read into the plan as the old one
   is today (the engines below the door do not change), the two execute calls over the
   existing dispatch, the info query over the existing plan fields, the per-thread error
   string beside the existing printer.
2. Every in-tree caller migrated, mechanically and gated: about 208 sources call
   `vfft_execute` (78 with the split signature), 84 set the thread count, 70 load a wisdom
   table, 74 read the route. No compatibility layer is kept (a clean library, not its
   history).
3. The API matrix gate rewritten against the new defaults, the policy gate where a law reads
   a default, and the two focused regressions (Rader, lane-major) run.
4. The documentation: the front door's design page and the performance pages' snippets.
