/* bench_vs_cufft.cu — cuFFT latency sweep for the VectorFFT comparison.
 *
 * Grid: K ∈ {8, 64, 256}, N ∈ {64, 256, 1024, 4096, 16384, 65536, 262144}.
 * For each cell, three measurements:
 *   1. compute-only       (data resident on GPU, just cufftExecZ2Z time)
 *   2. compute + D→H      (kernel + result transfer back to host)
 *   3. full round-trip    (H→D + kernel + D→H — drop-in CPU-FFT replacement use case)
 *
 * Output: CSV with columns N, K, compute_ns, comp_d2h_ns, roundtrip_ns
 * to stdout (redirect to file).
 *
 * Methodology: 5 warmup reps, then min-of-21 timing reps per measurement.
 * cudaDeviceSynchronize() before stopping each timer so we measure
 * completion, not just submission.
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <cufft.h>
#include <windows.h>

#define CUDA_CHECK(call) do { \
    cudaError_t e = call; \
    if (e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__, __LINE__, cudaGetErrorString(e)); \
        exit(1); \
    } \
} while (0)

#define CUFFT_CHECK(call) do { \
    cufftResult r = call; \
    if (r != CUFFT_SUCCESS) { \
        fprintf(stderr, "cuFFT error %s at %s:%d: %d\n", #call, __FILE__, __LINE__, r); \
        exit(1); \
    } \
} while (0)

static double now_ns(void) {
    LARGE_INTEGER f, c;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart * 1e9 / (double)f.QuadPart;
}

#define WARM 5
#define REPS 21

int main(void) {
    int Ns[] = {64, 256, 1024, 4096, 16384, 65536, 262144};
    int Ks[] = {8, 128, 256};
    int n_Ns = sizeof(Ns) / sizeof(Ns[0]);
    int n_Ks = sizeof(Ks) / sizeof(Ks[0]);

    /* Print device info to stderr (so stdout stays clean for CSV) */
    int dev = 0;
    CUDA_CHECK(cudaSetDevice(dev));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    fprintf(stderr, "[device] %s  SMs=%d  CC %d.%d  GlobalMem=%.1f GiB\n",
            prop.name, prop.multiProcessorCount, prop.major, prop.minor,
            (double)prop.totalGlobalMem / (1<<30));

    /* CSV header */
    printf("N,K,compute_ns,comp_d2h_ns,roundtrip_ns\n");

    /* Allocate host buffer big enough for the largest cell. */
    size_t max_NK = (size_t)Ns[n_Ns - 1] * Ks[n_Ks - 1];
    cufftDoubleComplex *h_buf = (cufftDoubleComplex *)malloc(max_NK * sizeof(cufftDoubleComplex));
    if (!h_buf) { fprintf(stderr, "host alloc failed\n"); return 1; }
    for (size_t i = 0; i < max_NK; i++) {
        h_buf[i].x = (double)rand() / RAND_MAX - 0.5;
        h_buf[i].y = (double)rand() / RAND_MAX - 0.5;
    }

    /* Allocate device buffer */
    cufftDoubleComplex *d_buf = nullptr;
    CUDA_CHECK(cudaMalloc(&d_buf, max_NK * sizeof(cufftDoubleComplex)));

    cudaEvent_t evt_start, evt_stop;
    CUDA_CHECK(cudaEventCreate(&evt_start));
    CUDA_CHECK(cudaEventCreate(&evt_stop));

    for (int ki = 0; ki < n_Ks; ki++) {
        int K = Ks[ki];
        for (int ni = 0; ni < n_Ns; ni++) {
            int N = Ns[ni];
            size_t NK = (size_t)N * K;
            size_t bytes = NK * sizeof(cufftDoubleComplex);

            /* Plan once per cell. cufftPlanMany with batch=K, stride=1,
             * dist=N — interleaved complex layout (matches the way users
             * typically batch in CUDA). */
            cufftHandle plan;
            int n_arr[1] = {N};
            CUFFT_CHECK(cufftPlanMany(&plan,
                /*rank*/      1, n_arr,
                /*inembed*/   NULL, /*istride*/ 1, /*idist*/ N,
                /*onembed*/   NULL, /*ostride*/ 1, /*odist*/ N,
                /*type*/      CUFFT_Z2Z, /*batch*/ K));

            /* ── Measurement 1: compute-only (data on device, kernel time) */
            CUDA_CHECK(cudaMemcpy(d_buf, h_buf, bytes, cudaMemcpyHostToDevice));
            for (int w = 0; w < WARM; w++)
                CUFFT_CHECK(cufftExecZ2Z(plan, d_buf, d_buf, CUFFT_FORWARD));
            CUDA_CHECK(cudaDeviceSynchronize());

            double best_compute = 1e30;
            for (int r = 0; r < REPS; r++) {
                CUDA_CHECK(cudaEventRecord(evt_start));
                CUFFT_CHECK(cufftExecZ2Z(plan, d_buf, d_buf, CUFFT_FORWARD));
                CUDA_CHECK(cudaEventRecord(evt_stop));
                CUDA_CHECK(cudaEventSynchronize(evt_stop));
                float ms;
                CUDA_CHECK(cudaEventElapsedTime(&ms, evt_start, evt_stop));
                double ns = ms * 1e6;
                if (ns < best_compute) best_compute = ns;
            }

            /* ── Measurement 2: compute + D→H (kernel + result transfer back) */
            for (int w = 0; w < WARM; w++) {
                CUFFT_CHECK(cufftExecZ2Z(plan, d_buf, d_buf, CUFFT_FORWARD));
                CUDA_CHECK(cudaMemcpy(h_buf, d_buf, bytes, cudaMemcpyDeviceToHost));
            }

            double best_comp_d2h = 1e30;
            for (int r = 0; r < REPS; r++) {
                double t0 = now_ns();
                CUFFT_CHECK(cufftExecZ2Z(plan, d_buf, d_buf, CUFFT_FORWARD));
                CUDA_CHECK(cudaMemcpy(h_buf, d_buf, bytes, cudaMemcpyDeviceToHost));
                double dt = now_ns() - t0;
                if (dt < best_comp_d2h) best_comp_d2h = dt;
            }

            /* ── Measurement 3: full round-trip (H→D + kernel + D→H) */
            for (int w = 0; w < WARM; w++) {
                CUDA_CHECK(cudaMemcpy(d_buf, h_buf, bytes, cudaMemcpyHostToDevice));
                CUFFT_CHECK(cufftExecZ2Z(plan, d_buf, d_buf, CUFFT_FORWARD));
                CUDA_CHECK(cudaMemcpy(h_buf, d_buf, bytes, cudaMemcpyDeviceToHost));
            }

            double best_roundtrip = 1e30;
            for (int r = 0; r < REPS; r++) {
                double t0 = now_ns();
                CUDA_CHECK(cudaMemcpy(d_buf, h_buf, bytes, cudaMemcpyHostToDevice));
                CUFFT_CHECK(cufftExecZ2Z(plan, d_buf, d_buf, CUFFT_FORWARD));
                CUDA_CHECK(cudaMemcpy(h_buf, d_buf, bytes, cudaMemcpyDeviceToHost));
                double dt = now_ns() - t0;
                if (dt < best_roundtrip) best_roundtrip = dt;
            }

            printf("%d,%d,%.0f,%.0f,%.0f\n",
                   N, K, best_compute, best_comp_d2h, best_roundtrip);
            fflush(stdout);

            CUFFT_CHECK(cufftDestroy(plan));
        }
    }

    cudaEventDestroy(evt_start);
    cudaEventDestroy(evt_stop);
    cudaFree(d_buf);
    free(h_buf);
    return 0;
}
