# The DFTI interface: MKL programs on VectorFFT

**Status (2026-10-16): built; its gate is `src/tools/gates/dfti_shim_gate.c`.**

A program written for Intel MKL's DFTI interface (`DftiCreateDescriptor`, `DftiSetValue`,
`DftiCommitDescriptor`, `DftiComputeForward`, ...) runs on VectorFFT by building against this
interface instead of MKL's. The interface is thin. It translates a committed descriptor into
`vfft.h` plans, delegates every transform to them, and never selects an engine, converts a
layout or reorders a spectrum itself. Whatever `vfft_create` serves through the front door is
served through DFTI in the geometries of section 3; whatever it refuses, the commit refuses
with the library's reason.

Scope: double precision, the interleaved layout, the C interface. The research behind it is
`docs/research/fftw_mkl_shim/REPORT.md` section 4 and `sources/dfti-surface.md`.

## 1. How a program switches

- **Source mode.** The interface is two files in `src/shims/dfti/`: the header, named
  `mkl_dfti.h`, and its implementation, `vfft_dfti.c`. A program puts that directory before
  MKL's on its include path and compiles `vfft_dfti.c` beside the library. A program that
  includes `mkl_dfti.h` builds unchanged.
- **The umbrella header.** A program that includes MKL's `mkl.h` receives MKL's own DFTI
  declarations. It replaces that include with `mkl_dfti.h` and the other MKL headers it uses.
- **Our own symbols.** The functions are named `vfft_dfti_*`, and the header maps the DFTI
  names onto them. MKL can stay linked beside the interface for BLAS or LAPACK, and no symbol
  collides. Relinking an already-built program against the interface under MKL's own names is
  not provided.
- **Constants.** The numeric values of the DFTI parameters, values and status codes are
  MKL's. They are interoperability facts, recorded in the research. The header's text is our
  own, written from Intel's public reference.
- **`MKL_LONG`.** `long int` by default, 64 bits under `MKL_ILP64`. Variable arguments are
  read at MKL's widths: enumerations as 32-bit `int`, integers as `MKL_LONG`, scales as
  `double`, lengths and strides as `MKL_LONG *`, names as `char *`.
- **License.** MIT, as the library.

## 2. The descriptor

`DftiCreateDescriptor` fixes the precision, the forward domain, the rank and the lengths. A
single-precision descriptor returns `DFTI_UNIMPLEMENTED` (6). Ranks 1 to 7 are accepted at
create, as MKL accepts them (0 or 8 returns 2); which ranks are served is the commit's answer.

| Parameter | Accepted | Effect |
|---|---|---|
| `FORWARD_SCALE`, `BACKWARD_SCALE` | any `double` (default 1) | applied after compute, section 5 |
| `NUMBER_OF_TRANSFORMS` | >= 1 (default 1) | the batch count K |
| `COMPLEX_STORAGE` | `COMPLEX_COMPLEX` | `REAL_REAL` returns 6: split data is served through `vfft.h` |
| `REAL_STORAGE` | `REAL_REAL` | its only value; `REAL_COMPLEX` returns 6, as on MKL |
| `CONJUGATE_EVEN_STORAGE` | `COMPLEX_COMPLEX` | `COMPLEX_REAL` and `REAL_REAL` return 6 |
| `PACKED_FORMAT` | `CCE` | `CCS`, `PACK`, `PERM` return 6 |
| `PLACEMENT` | `INPLACE` (default), `NOT_INPLACE` | the plan's placement |
| `INPUT_STRIDES`, `OUTPUT_STRIDES`, `INPUT_DISTANCE`, `OUTPUT_DISTANCE` | any, as MKL stores them | read at commit, section 3 |
| `ORDERING` | `ORDERED`; `BACKWARD_SCRAMBLED` | scrambled is accepted and served in natural order, with one printed line |
| `DESTROY_INPUT` | `AVOID` (default), `ALLOW` | `ALLOW` sets `VFFT_FLAG_DESTROY_INPUT` |
| `WORKSPACE` | `ALLOW`, `AVOID` | no effect |
| `NUMBER_OF_USER_THREADS` | >= 1 (default 1) | the number of plans the commit builds, section 4 |
| `THREAD_LIMIT` | >= 0 (default 0) | each plan's thread count, section 4 |
| `DESCRIPTOR_NAME` | a string | stored and read back |
| `COMMIT_STATUS`, `VERSION` | read only | `VERSION` also answers a NULL handle |
| `FORWARD_SIGN`, `INITIALIZATION_EFFORT`, `FORWARD_ORDERING`, `BACKWARD_ORDERING`, `TRANSPOSE` (set), `FWD_DISTANCE`, `BWD_DISTANCE` | none | 3, as on MKL |

The lifecycle is MKL's:

- Any `DftiSetValue` after commit uncommits, even with the same value. The plans stay until
  the next commit or the free.
- A compute on an uncommitted descriptor returns 2 and computes nothing.
- A committed descriptor's copy is committed, with its own plans built from the same
  configuration (a wisdom replay). An uncommitted descriptor's copy is uncommitted.
- `DftiFreeDescriptor` frees the plans and sets the handle to NULL.
- A NULL handle returns 5 everywhere, except the `VERSION` query.
- The defaults are MKL's measured ones, including the real-domain pitch on the output strides
  of a real descriptor.

## 3. Commit

The commit translates the descriptor into one `vfft_config_t` per direction it serves: layout
`INTERLEAVED`, order `NATURAL`, batch `CONTIGUOUS` or `LANE_MAJOR`. It then creates the plans.
The first element offset (`s0` of the strides) is not part of the configuration; compute adds
it to the pointer.

**Complex domain** (strides and distances in complex elements; `n1 ... nd` are the lengths,
slowest first; `P = n1 ... nd`):

| Form | Strides, in and out | Distance | Configuration |
|---|---|---|---|
| one transform | tight: `{s0, n2...nd, ..., nd, 1}` | not read | K = 1 |
| contiguous batch | tight | P | `VFFT_BATCH_CONTIGUOUS` |
| lane-major batch | K times tight: `{s0, K n2...nd, ..., K nd, K}` | 1 | `VFFT_BATCH_LANE_MAJOR` |

In place, the output strides and distance are not read, as on MKL.

**Real domain** (the reals in real elements, the half spectrum in complex elements;
`H = nd/2 + 1`):

| Form | Real side | Complex side | Distance, real / complex | Configuration |
|---|---|---|---|---|
| out of place | tight | packed: `{s0, ..., H, 1}` | not read at K = 1 | K = 1 |
| in place, rank 1 | `{0, 1}` | `{0, 1}` | not read | one plane of 2H doubles |
| in place, rank 2 and 3 | padded: `{0, ..., 2H, 1}` | `{0, ..., H, 1}` | not read | padded rows |
| contiguous batch, out of place | tight | packed | P / `n1...n(d-1) H` | `VFFT_BATCH_CONTIGUOUS` |
| contiguous batch, in place, rank 1 | `{0, 1}` | `{0, 1}` | 2H / H | `VFFT_BATCH_CONTIGUOUS` |
| lane-major batch, rank 1 | `{s0, K}` | `{s0, K}` | 1 / 1 | `VFFT_BATCH_LANE_MAJOR`, both placements |
| lane-major batch, rank 2, out of place | `{s0, N2 K, K}` | `{s0, H K, K}` | 1 / 1 | `VFFT_BATCH_LANE_MAJOR` |
| lane-major batch, rank 2, in place | `{0, 2H K, K}` | `{0, H K, K}` | 1 / 1 | `VFFT_BATCH_LANE_MAJOR` |

Any other strides or distances return 6, with one printed line naming the served forms. Those
include padded or arbitrary pitches, negative strides and overlapping data. Whether the
library serves a recognised form at the given lengths, rank and batch is `vfft_create`'s
answer. Its refusal returns 6, with the library's reason (`vfft_error()`) printed.

**Strides belong to the call.** MKL's contract: forward reads the real side through the input
strides and writes the complex side through the output strides; backward reverses them. A
program swaps the strides and recommits between directions. The commit builds the R2C plan
when the committed strides fit the forward, and the C2R plan when they fit the backward. At
rank 1 with the default strides both fit. At higher ranks one fits per commit. A compute in a
direction the committed strides do not fit returns 3, with a printed line saying to swap and
recommit. MKL returns 0 there and writes garbage.

**MKL's own refusals are kept.** A batch with no distance returns 3. An in-place real
transform of rank 2 or more with the default strides returns 3. MKL's default output strides
for an out-of-place real transform of rank 2 or more (the real pitch, not the packed one) are
not a served form: 6, with a line naming the packed strides.

**A recommit keeps the plans** when the translated configurations equal the committed ones. A
program that sets a value and recommits per call pays nothing more.

**Cost.** A commit that misses wisdom measures the cell's candidates (seconds at a small cell,
minutes at the largest) and saves the winner. Every later commit of the cell on the
machine replays it. Like `vfft_create`, the commit leaves flush-to-zero and
denormals-are-zero on in the committing thread's floating-point mode.

## 4. Threads

- **`THREAD_LIMIT` > 0:** each plan runs on that many threads.
- **`THREAD_LIMIT` = 0, "as many as available":** each plan runs on the machine's P-cores
  (`vfft_info().cores`).
  When `NUMBER_OF_USER_THREADS` > 1, each plan runs on one thread instead, since the program's
  own threads are the parallelism. The thread-count ladder (`nthreads = AUTO`) takes over this
  choice when it lands.
- **After commit,** `THREAD_LIMIT` reads back the plans' thread count, as MKL reports its
  actual count.
- **One descriptor, many threads.** DFTI allows one committed descriptor to be computed from
  several threads at once; a `vfft.h` plan runs one execute at a time. The commit therefore
  builds `NUMBER_OF_USER_THREADS` plans: the first may measure, the rest replay its wisdom. A
  compute borrows a free plan (one atomic compare-and-swap when uncontended) and returns it.
  When every plan is busy the compute waits for one, and the first wait on a descriptor prints
  one line naming `DFTI_NUMBER_OF_USER_THREADS`. Results are always correct, and throughput
  scales with the declared count.
- **Different descriptors** may compute at once. Their threaded passes take turns on the
  library's pool, never corrupting each other.

## 5. Compute

`DftiComputeForward` and `DftiComputeBackward` read exactly the committed number of pointers:
one in place, two out of place. They add the offsets, borrow a plan, call
`vfft_execute(plan, direction, in, out)`, and return 0.

If the direction's scale is not 1, the compute then multiplies every double the plan wrote
(`vfft_info().out_doubles`) by it, on the calling thread. The result equals MKL's within
rounding, at the price of one more pass over the output. That pass disappears when the library
gains fused scaling, scaled final writers bound at plan time, after the release.

## 6. Errors and messages

| Status | Returned for |
|---|---|
| 0 `DFTI_NO_ERROR` | success |
| 1 `DFTI_MEMORY_ERROR` | memory exhausted |
| 2 `DFTI_INVALID_CONFIGURATION` | invalid create arguments; a compute on an uncommitted descriptor |
| 3 `DFTI_INCONSISTENT_CONFIGURATION` | a parameter that cannot be set, or a value outside the vocabulary; MKL's inconsistent combinations; a compute in a direction the committed strides do not fit |
| 5 `DFTI_BAD_DESCRIPTOR` | a NULL handle |
| 6 `DFTI_UNIMPLEMENTED` | a legitimate setting or geometry VectorFFT does not serve; a `vfft_create` refusal |

- `DftiErrorClass(s, c)` is `s == c`.
- `DftiErrorMessage` returns a static string in our own words, at most 80 characters.
- Printed lines go to stderr, prefixed `vfft dfti:`. They cover refusals with their reason,
  the scrambled-order substitution, a compute in an unfit direction and the first wait on a
  shared descriptor. Each distinct line is printed once per process.

## 7. Where the interface differs from MKL

- A compute in a direction the committed strides do not fit returns 3. MKL returns 0 with
  wrong output.
- A first commit of a cell on a machine measures, from seconds to minutes. MKL's commit
  is analytic.
- A scale other than 1 costs one pass over the output.
- `BACKWARD_SCRAMBLED` prints that natural order is served. MKL serves natural order silently.
- The commit leaves flush-to-zero and denormals-are-zero on in the committing thread.
- A copy of an uncommitted descriptor reports `UNCOMMITTED`. MKL reports 0.

## 8. Not served

- Single precision.
- Ranks 5 to 7.
- Split storage: `COMPLEX_STORAGE` or `CONJUGATE_EVEN_STORAGE` set to `REAL_REAL`. `vfft.h`
  serves split data directly.
- The packed real formats CCS, PACK and PERM.
- Strides and distances outside the forms of section 3.
- Fortran entry points.
- The cluster, offload and SYCL interfaces.
- Relinking an already-built program under MKL's symbol names.

## 9. Files and verification

- `src/shims/dfti/mkl_dfti.h`: the header.
- `src/shims/dfti/vfft_dfti.c`: the implementation. It includes `vfft.h` and nothing from
  `src/core`.
- `src/tools/gates/dfti_shim_gate.c`: the gate. It checks every form of section 3 in both
  directions against a naive DFT. It also covers:
  - scales;
  - the statuses: an uncommitted compute 2, a NULL handle 5, the NULL-handle `VERSION`
    query 0, refusals 6 and 3;
  - copies;
  - a recommit with an unchanged configuration keeping its plans;
  - one descriptor computed from eight threads, with `NUMBER_OF_USER_THREADS` set to 8
    and left at 1.
- Intel's DFTI examples (MKL's installed examples, run locally only, since they are Intel's
  copyright): each builds against the interface's header beside MKL's service header and
  verifies its own result. The double-precision ones pass, except a contiguous batch of 3D
  transforms, which the library does not serve.

