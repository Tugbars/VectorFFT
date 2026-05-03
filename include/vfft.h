/**
 * vfft.h — VectorFFT public C API
 *
 * Opaque-handle interface for the VectorFFT library.
 * All internal structures are hidden behind vfft_plan handles.
 *
 * Data layout: split-complex (separate re[] and im[] arrays).
 *   Complex: re[n * K + k], im[n * K + k]  for n=0..N-1, k=0..K-1
 *   Real:    re[n * K + k]                  for n=0..N-1, k=0..K-1
 *
 * Convention: bwd(fwd(x)) = N * x  (unnormalized backward).
 *   Use vfft_execute_bwd_normalized for bwd(fwd(x)) = x.
 *
 * Output order: the forward transform produces output in implementation-defined
 *   (digit-reversed) order, NOT natural DFT order. This is by design — the
 *   DIT forward and DIF backward permutations cancel, giving a permutation-free
 *   roundtrip with zero reordering overhead.
 *
 *   Implications:
 *   - Roundtrip (fwd then bwd) is exact — no permutation needed.
 *   - Pointwise multiply of two spectra from the SAME plan is correct
 *     (both are in the same scrambled order) — convolution works.
 *   - Reading individual frequency bins (e.g., DC, Nyquist) from forward
 *     output requires knowing the permutation order.
 *   - R2C output IS in natural order (the postprocess unscrambles it).
 *   - 2D forward output is in natural order along axis-1 (rows) but
 *     digit-reversed along axis-0 (columns).
 *
 *   Future: vfft_permute(plan, re, im) will be provided to reorder forward
 *   output into natural DFT order for users who need frequency-domain
 *   inspection, filtering, or spectral analysis. The permutation table is
 *   already computed internally — exposing it is a minor addition.
 *
 * Thread safety:
 *   - Plans are immutable after creation — safe to share across threads.
 *   - vfft_execute_* modifies only the user's data buffers (re, im).
 *     No global or plan state is written during execution.
 *   - vfft_set_num_threads is global — call once before creating plans.
 *     Do not call concurrently with execute or plan creation.
 *   - vfft_init / vfft_plan_* allocate memory — call from one thread.
 *
 * Threading model (two options, do not mix):
 *
 *   Option A — internal parallelism (recommended for single-stream):
 *     vfft_set_num_threads(8);        // library parallelizes internally
 *     vfft_execute_fwd(plan, re, im); // call from ONE thread
 *
 *   Option B — external parallelism (recommended for multi-stream):
 *     vfft_set_num_threads(1);        // disable internal threading
 *     // Each of your threads calls execute on its own data:
 *     //   thread 0: vfft_execute_fwd(plan, re0, im0);
 *     //   thread 1: vfft_execute_fwd(plan, re1, im1);
 *     // Safe: plan is read-only, no shared mutable state at T=1.
 *
 *   Do not combine both (calling execute from multiple threads with
 *   internal threading enabled) — this oversubscribes the CPU and the
 *   internal thread pool is not designed for concurrent callers.
 *
 * === Quick start ===
 *
 *   #include <vfft.h>
 *
 *   vfft_init();
 *   vfft_plan p = vfft_plan_c2c(1024, 256);
 *   double *re = vfft_alloc(1024 * 256 * sizeof(double));
 *   double *im = vfft_alloc(1024 * 256 * sizeof(double));
 *   // ... fill re[], im[] ...
 *   vfft_execute_fwd(p, re, im);
 *   vfft_execute_bwd_normalized(p, re, im);  // roundtrip: output == input
 *   vfft_destroy(p);
 *   vfft_free(re); vfft_free(im);
 *
 * === CMake integration ===
 *
 *   # In your CMakeLists.txt:
 *   find_package(VectorFFT REQUIRED)
 *   target_link_libraries(myapp PRIVATE VectorFFT::vfft)
 *
 *   # Build and install VectorFFT:
 *   git clone https://github.com/user/VectorFFT.git
 *   cd VectorFFT && mkdir build && cd build
 *   cmake .. -DCMAKE_BUILD_TYPE=Release
 *   cmake --build . --config Release
 *   cmake --install . --prefix /usr/local  # or any prefix
 *
 *   # Then in your project:
 *   cmake -DCMAKE_PREFIX_PATH=/usr/local ..
 *
 * === pkg-config ===
 *
 *   pkg-config --cflags --libs vectorfft
 */
#ifndef VFFT_H
#define VFFT_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ═══════════════════════════════════════════════════════════════
 * OPAQUE HANDLE
 * ═══════════════════════════════════════════════════════════════ */

/** Opaque FFT plan handle. */
typedef struct vfft_plan_s *vfft_plan;

/* ═══════════════════════════════════════════════════════════════
 * PLAN CREATION FLAGS
 *
 * Mirrors FFTW's flag conventions. Pass to vfft_plan_* as the trailing
 * flags argument. Combine with bitwise OR (e.g. VFFT_MEASURE|VFFT_WISDOM_ONLY).
 * ═══════════════════════════════════════════════════════════════ */

/** Heuristic plan via cost model. Fast plan creation, possibly suboptimal.
 *  Equivalent to FFTW_ESTIMATE. */
#define VFFT_ESTIMATE        0u

/** Wisdom-driven plan. On wisdom miss, runs a per-cell calibration
 *  and inserts the result into the in-memory wisdom database (so subsequent
 *  identical (N, K) plans hit the cache). Equivalent to FFTW_MEASURE. */
#define VFFT_MEASURE         (1u << 0)

/** Wider per-cell calibration on miss (more candidates, more reps).
 *  Slower plan creation; usually picks the same plan as MEASURE.
 *  Equivalent to FFTW_EXHAUSTIVE. */
#define VFFT_EXHAUSTIVE      (1u << 1)

/** Use only what is already in the wisdom database. Returns NULL on miss
 *  (no calibration). Useful when plan creation must be fast and predictable.
 *  Equivalent to FFTW_WISDOM_ONLY. */
#define VFFT_WISDOM_ONLY     (1u << 2)

/* ═══════════════════════════════════════════════════════════════
 * WISDOM LIFECYCLE
 *
 * Wisdom is a measured-plan cache. The library maintains an in-memory
 * wisdom database initialized empty by vfft_init(). Users opt in by
 * loading a wisdom file (their own calibration output, or a sample shipped
 * under examples/<arch>/wisdom.txt). MEASURE/EXHAUSTIVE plans consult
 * the database; on miss, they calibrate that cell and insert the result.
 *
 * The library does NOT auto-load wisdom from disk. Users with no wisdom
 * loaded who pass VFFT_MEASURE will pay one-time calibration per cell
 * (matching FFTW_MEASURE behavior).
 *
 * Wisdom is per-precision and per-microarchitecture. The sample wisdom
 * under examples/14900KF/wisdom.txt was generated on Intel i9-14900KF
 * (Raptor Lake, AVX2, single-thread); it may be sub-optimal on other
 * hardware. Generate your own with vfft_save_wisdom() after exercising
 * the plans you care about under VFFT_MEASURE.
 * ═══════════════════════════════════════════════════════════════ */

/**
 * vfft_load_wisdom — Load wisdom from a file, merging into the in-memory
 * database. Existing entries with matching (N, K) are replaced.
 * @return  0 on success, non-zero if the file was unreadable or malformed.
 */
int vfft_load_wisdom(const char *path);

/**
 * vfft_save_wisdom — Save the in-memory wisdom database to a file.
 * @return  0 on success, non-zero on I/O error.
 */
int vfft_save_wisdom(const char *path);

/**
 * vfft_forget_wisdom — Discard the in-memory wisdom database.
 * Subsequent VFFT_MEASURE plans will calibrate from scratch.
 */
void vfft_forget_wisdom(void);

/* ═══════════════════════════════════════════════════════════════
 * INITIALIZATION
 * ═══════════════════════════════════════════════════════════════ */

/**
 * vfft_init — Initialize the library.
 *
 * Sets FTZ/DAZ for optimal floating-point performance.
 * Call once at program start, before any other vfft_* calls.
 * Automatically called by plan creation if not called explicitly.
 */
void vfft_init(void);

/**
 * vfft_pin_thread — Pin the calling thread to a specific CPU core.
 *
 * Useful for benchmarking on hybrid CPUs (P-core vs E-core).
 * @param core_id  0-based logical processor index.
 * @return         0 on success, -1 on failure.
 */
int vfft_pin_thread(int core_id);

/* ═══════════════════════════════════════════════════════════════
 * PLAN CREATION
 * ═══════════════════════════════════════════════════════════════ */

/**
 * vfft_plan_c2c — Create a complex-to-complex FFT plan.
 *
 * @param N   Transform size (any positive integer; highly composite N is fastest).
 * @param K   Batch count (number of independent transforms). K >= 4 for SIMD.
 * @return    Plan handle, or NULL on failure.
 *
 * The plan is reusable: call vfft_execute_* many times with different data.
 * Uses the heuristic planner. For calibrated plans, use vfft_plan_c2c_measure.
 */
vfft_plan vfft_plan_c2c(int N, size_t K);

/**
 * vfft_plan_c2c_measure — Create a calibrated complex-to-complex FFT plan.
 *
 * Tries multiple factorizations and measures actual performance.
 * Slower to create but may produce faster plans for repeated execution.
 *
 * @param N   Transform size.
 * @param K   Batch count.
 * @return    Plan handle, or NULL on failure.
 */
vfft_plan vfft_plan_c2c_measure(int N, size_t K);

/**
 * vfft_plan_r2c — Create a real-to-complex FFT plan.
 *
 * N-point real FFT → N/2+1 complex output (Hermitian symmetry).
 * N must be even.
 *
 * @param N   Transform size (must be even).
 * @param K   Batch count.
 * @return    Plan handle, or NULL on failure.
 */
vfft_plan vfft_plan_r2c(int N, size_t K);

/**
 * vfft_plan_2d — Create a 2D complex FFT plan.
 *
 * In-place, row-major layout: re[i*N2 + j], im[i*N2 + j].
 * bwd(fwd(x)) = N1*N2 * x.
 *
 * @param N1   Number of rows (axis-0 FFT length).
 * @param N2   Number of columns (axis-1 FFT length).
 * @return     Plan handle, or NULL on failure.
 */
vfft_plan vfft_plan_2d(int N1, int N2);

/**
 * vfft_plan_2d_r2c — Create a 2D real-to-complex FFT plan.
 *
 * Forward: N1*N2 reals -> N1*(N2/2+1) complex bins (reduces along inner axis).
 * Backward: reverse, scaled bwd(fwd(x)) = N1*N2 * x.
 *
 * @param N1   Number of rows (axis-0).
 * @param N2   Number of columns (axis-1, must be even).
 * @return     Plan handle, or NULL on failure.
 */
vfft_plan vfft_plan_2d_r2c(int N1, int N2);

/**
 * vfft_plan_dct2 — Create a DCT-II / DCT-III plan (FFTW REDFT10 / REDFT01).
 *
 * One plan handles both directions:
 *   vfft_execute_dct2 = forward (REDFT10)
 *   vfft_execute_dct3 = backward (REDFT01); inverse up to scale 2N.
 *
 * Constraint: N must be even.
 */
vfft_plan vfft_plan_dct2(int N, size_t K);

/**
 * vfft_plan_dct4 — Create a DCT-IV plan (FFTW REDFT11).
 *
 * Involutory up to scale 2N: dct4(dct4(x))/(2N) = x. One plan, one executor.
 *
 * Constraint: N must be even.
 */
vfft_plan vfft_plan_dct4(int N, size_t K);

/**
 * vfft_plan_dst2 — Create a DST-II / DST-III plan (FFTW RODFT10 / RODFT01).
 *
 * One plan handles both directions:
 *   vfft_execute_dst2 = forward (RODFT10)
 *   vfft_execute_dst3 = backward (RODFT01); inverse up to scale 2N.
 *
 * Constraint: N must be even.
 */
vfft_plan vfft_plan_dst2(int N, size_t K);

/**
 * vfft_plan_dht — Create a Discrete Hartley Transform plan (FFTW_DHT).
 *
 * Self-inverse up to scale 1/N: dht(dht(x))/N = x. One plan, one executor.
 *
 * Constraint: N must be even.
 */
vfft_plan vfft_plan_dht(int N, size_t K);

/**
 * vfft_destroy — Destroy a plan and free all resources.
 *
 * @param p   Plan handle (NULL is safe — no-op).
 */
void vfft_destroy(vfft_plan p);

/* ═══════════════════════════════════════════════════════════════
 * EXECUTION — COMPLEX-TO-COMPLEX
 * ═══════════════════════════════════════════════════════════════ */

/**
 * vfft_execute_fwd — Forward (DIT) complex FFT, in-place.
 *
 * @param p    Plan from vfft_plan_c2c.
 * @param re   Real parts: re[n*K + k], N*K doubles. Must be 64-byte aligned.
 * @param im   Imaginary parts: same layout. Must be 64-byte aligned.
 */
void vfft_execute_fwd(vfft_plan p, double *re, double *im);

/**
 * vfft_execute_bwd — Backward (DIF) complex FFT, in-place, unnormalized.
 *
 * Output = N * IFFT(input). Divide by N to get true inverse.
 */
void vfft_execute_bwd(vfft_plan p, double *re, double *im);

/**
 * vfft_execute_bwd_normalized — Backward complex FFT with 1/N scaling.
 *
 * bwd_normalized(fwd(x)) = x.
 */
void vfft_execute_bwd_normalized(vfft_plan p, double *re, double *im);

/* ═══════════════════════════════════════════════════════════════
 * EXECUTION — REAL-TO-COMPLEX / COMPLEX-TO-REAL
 * ═══════════════════════════════════════════════════════════════ */

/**
 * vfft_execute_r2c — Forward real-to-complex FFT.
 *
 * @param p         Plan from vfft_plan_r2c.
 * @param real_in   Real input: real_in[n*K + k], N*K doubles.
 * @param out_re    Complex output real parts: must be N*K doubles (used as workspace).
 *                  Only the first (N/2+1)*K contain valid output.
 * @param out_im    Complex output imag parts: (N/2+1)*K doubles.
 */
void vfft_execute_r2c(vfft_plan p, const double *real_in,
                       double *out_re, double *out_im);

/**
 * vfft_execute_c2r — Backward complex-to-real FFT.
 *
 * Output = N * original_input. Divide by N to normalize.
 *
 * @param p         Plan from vfft_plan_r2c.
 * @param in_re     Complex input real parts: (N/2+1)*K doubles.
 * @param in_im     Complex input imag parts: (N/2+1)*K doubles.
 * @param real_out  Real output: N*K doubles.
 */
void vfft_execute_c2r(vfft_plan p, const double *in_re, const double *in_im,
                       double *real_out);

/* ═══════════════════════════════════════════════════════════════
 * EXECUTION — 2D REAL-TO-COMPLEX / COMPLEX-TO-REAL
 * ═══════════════════════════════════════════════════════════════ */

/**
 * vfft_execute_2d_r2c — 2D forward real-to-complex FFT.
 *
 * @param p         Plan from vfft_plan_2d_r2c.
 * @param real_in   Real input: real_in[i*N2 + j], N1*N2 doubles.
 * @param out_re    Complex output Re bins: N1*(N2/2+1) doubles.
 * @param out_im    Complex output Im bins: N1*(N2/2+1) doubles.
 */
void vfft_execute_2d_r2c(vfft_plan p, const double *real_in,
                          double *out_re, double *out_im);

/**
 * vfft_execute_2d_c2r — 2D backward complex-to-real FFT.
 *
 * Output = N1*N2 * original_input. Divide by N1*N2 to normalize.
 */
void vfft_execute_2d_c2r(vfft_plan p, const double *in_re, const double *in_im,
                          double *real_out);

/* ═══════════════════════════════════════════════════════════════
 * EXECUTION — DCT / DST / DHT (real-to-real)
 *
 * All r2r layouts: in[n*K + k] for n=0..N-1, k=0..K-1; out same shape.
 * In-place safe (caller may pass in == out).
 * Conventions match FFTW: unnormalized; user divides for roundtrip.
 * ═══════════════════════════════════════════════════════════════ */

/**
 * vfft_execute_dct2 — DCT-II (FFTW REDFT10).
 *   Y[k] = 2 * sum_{n=0..N-1} x[n] * cos(pi*k*(2n+1)/(2N))
 */
void vfft_execute_dct2(vfft_plan p, const double *in, double *out);

/**
 * vfft_execute_dct3 — DCT-III (FFTW REDFT01), inverse of DCT-II up to scale 2N.
 *   For y = dct2(x), x_recovered = dct3(y) / (2N).
 */
void vfft_execute_dct3(vfft_plan p, const double *in, double *out);

/**
 * vfft_execute_dct4 — DCT-IV (FFTW REDFT11). Involutory up to scale 2N.
 *   For y = dct4(x), x_recovered = dct4(y) / (2N).
 */
void vfft_execute_dct4(vfft_plan p, const double *in, double *out);

/**
 * vfft_execute_dst2 — DST-II (FFTW RODFT10).
 */
void vfft_execute_dst2(vfft_plan p, const double *in, double *out);

/**
 * vfft_execute_dst3 — DST-III (FFTW RODFT01), inverse of DST-II up to scale 2N.
 */
void vfft_execute_dst3(vfft_plan p, const double *in, double *out);

/**
 * vfft_execute_dht — Discrete Hartley Transform (FFTW_DHT). Self-inverse up to 1/N.
 *   For y = dht(x), x_recovered = dht(y) / N.
 */
void vfft_execute_dht(vfft_plan p, const double *in, double *out);

/* ═══════════════════════════════════════════════════════════════
 * THREADING
 * ═══════════════════════════════════════════════════════════════ */

/**
 * vfft_set_num_threads — Set the number of threads for FFT execution.
 *
 * @param n   Number of threads (1 = single-threaded, default).
 *            Call before creating plans.
 */
void vfft_set_num_threads(int n);

/**
 * vfft_get_num_threads — Query current thread count.
 */
int vfft_get_num_threads(void);

/* ═══════════════════════════════════════════════════════════════
 * MEMORY
 * ═══════════════════════════════════════════════════════════════ */

/**
 * vfft_alloc — Allocate 64-byte aligned memory for FFT data.
 *
 * @param bytes   Size in bytes.
 * @return        Aligned pointer, or NULL on failure. Free with vfft_free.
 */
void *vfft_alloc(size_t bytes);

/**
 * vfft_free — Free memory allocated by vfft_alloc.
 */
void vfft_free(void *p);

/* ═══════════════════════════════════════════════════════════════
 * INTERLEAVED CONVERSION
 * ═══════════════════════════════════════════════════════════════ */

/**
 * vfft_deinterleave — Convert {re,im,re,im,...} to separate re[],im[].
 *
 * @param interleaved   Input: 2*count doubles in {re,im} pairs.
 * @param re            Output: count doubles (real parts).
 * @param im            Output: count doubles (imaginary parts).
 * @param count         Number of complex elements.
 */
void vfft_deinterleave(const double *interleaved, double *re, double *im, size_t count);

/**
 * vfft_reinterleave — Convert separate re[],im[] to {re,im,re,im,...}.
 *
 * @param re            Input: count doubles (real parts).
 * @param im            Input: count doubles (imaginary parts).
 * @param interleaved   Output: 2*count doubles in {re,im} pairs.
 * @param count         Number of complex elements.
 */
void vfft_reinterleave(const double *re, const double *im, double *interleaved, size_t count);

/* ═══════════════════════════════════════════════════════════════
 * QUERY
 * ═══════════════════════════════════════════════════════════════ */

/**
 * vfft_version — Return version string (e.g., "0.1.0").
 */
const char *vfft_version(void);

/**
 * vfft_isa — Return ISA name used at compile time ("avx2", "avx512", "scalar").
 */
const char *vfft_isa(void);

#ifdef __cplusplus
}
#endif

#endif /* VFFT_H */
