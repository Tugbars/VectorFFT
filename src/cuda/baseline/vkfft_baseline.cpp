// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tuğbars Heptaşkın
//
// vkfft_baseline.cpp -- the baseline this project measures itself against.
//
// Times VkFFT's CUDA backend on 1D complex-to-complex single-precision in-place
// transforms of power-of-two length, batched so every size moves the same number
// of bytes. Next to each transform it times a plain in-place copy kernel on the
// same buffer: the copy reads every element once and writes it once, which is the
// least memory traffic a one-pass transform can have.
//
// One round takes four samples. A sample queues several launches and then
// synchronizes once, so the cost of a launch and of the synchronization is spread
// over the queue (a lone launch costs about 20 us on this driver, which would
// swamp a small buffer):
//   - forward + inverse pairs (the inverse is normalized, so the data stay bounded)
//   - the copy with 8 bytes per thread
//   - the copy with 32 bytes per thread
//   - one forward and one inverse on their own, each synchronized (tells the two
//     directions apart; only meaningful when a transform takes about 1 ms or more)
// Medians over the rounds are reported with the spread.
//
// Build: see ../CMakeLists.txt (needs -DVKFFT_ROOT=<VkFFT checkout>).

#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS
#endif
#include <cuda.h>
#include <cuda_runtime.h>
#include <nvrtc.h>
#include <cuComplex.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "vkFFT.h"

namespace {

struct Options {
    int min_log2 = 3;
    int max_log2 = 16;
    int total_log2 = 26;       // complex elements in the buffer (2^26 * 8 B = 512 MB)
    int rounds = 31;
    int warmup = 8;
    double queue_ms = 10.0;    // a queued sample is sized to last about this long
    bool forward_only = false; // profile mode: lone forward launches, no copy kernels
    bool check = true;
    std::string csv;
    std::string dump_dir;      // write VkFFT's generated source and binary per size
};

double seconds_now() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

struct Stat {
    double med = 0, min = 0, p10 = 0, p90 = 0;
};

Stat stat_of(std::vector<double> v) {
    Stat s;
    if (v.empty()) return s;
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    s.min = v.front();
    s.med = v[n / 2];
    s.p10 = v[(n - 1) / 10];
    s.p90 = v[((n - 1) * 9) / 10];
    return s;
}

void die(const char* what, long code) {
    std::fprintf(stderr, "FAILED: %s (code %ld)\n", what, code);
    std::exit(1);
}

// ---------------------------------------------------------------------------
// The copy kernels (memory reference). Each swaps re and im in place, so every
// byte is read once and written once and the values keep their size.
// ---------------------------------------------------------------------------
const char* kCopySource = R"(
extern "C" __global__ void copy8(float2* buf, unsigned long long n) {
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float2 v = buf[i];
        float2 r; r.x = v.y; r.y = v.x;
        buf[i] = r;
    }
}
extern "C" __global__ void copy32(float4* buf, unsigned long long n) {
    unsigned long long i = ((unsigned long long)blockIdx.x * blockDim.x + threadIdx.x) * 2ull;
    if (i + 1ull < n) {
        float4 a = buf[i];
        float4 b = buf[i + 1ull];
        float4 ra; ra.x = a.y; ra.y = a.x; ra.z = a.w; ra.w = a.z;
        float4 rb; rb.x = b.y; rb.y = b.x; rb.z = b.w; rb.w = b.z;
        buf[i] = ra;
        buf[i + 1ull] = rb;
    }
}
)";

struct CopyKernels {
    CUmodule module = nullptr;
    CUfunction copy8 = nullptr;
    CUfunction copy32 = nullptr;
};

CopyKernels build_copy_kernels(CUdevice dev) {
    int major = 0, minor = 0;
    cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev);
    cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev);
    char arch[64];
    std::snprintf(arch, sizeof arch, "--gpu-architecture=sm_%d%d", major, minor);
    const char* opts[] = {arch};

    nvrtcProgram prog;
    nvrtcResult r = nvrtcCreateProgram(&prog, kCopySource, "copy.cu", 0, nullptr, nullptr);
    if (r != NVRTC_SUCCESS) die("nvrtcCreateProgram", r);
    r = nvrtcCompileProgram(prog, 1, opts);
    if (r != NVRTC_SUCCESS) {
        size_t log_size = 0;
        nvrtcGetProgramLogSize(prog, &log_size);
        std::string log(log_size, '\0');
        nvrtcGetProgramLog(prog, log.data());
        std::fprintf(stderr, "%s\n", log.c_str());
        die("nvrtcCompileProgram(copy)", r);
    }
    size_t size = 0;
    nvrtcGetCUBINSize(prog, &size);
    std::vector<char> image(size);
    nvrtcGetCUBIN(prog, image.data());
    nvrtcDestroyProgram(&prog);

    CopyKernels k;
    CUresult c = cuModuleLoadDataEx(&k.module, image.data(), 0, nullptr, nullptr);
    if (c != CUDA_SUCCESS) die("cuModuleLoadDataEx(copy)", c);
    c = cuModuleGetFunction(&k.copy8, k.module, "copy8");
    if (c != CUDA_SUCCESS) die("cuModuleGetFunction(copy8)", c);
    c = cuModuleGetFunction(&k.copy32, k.module, "copy32");
    if (c != CUDA_SUCCESS) die("cuModuleGetFunction(copy32)", c);
    return k;
}

void launch(CUfunction fn, unsigned grid, unsigned block, void** args) {
    CUresult c = cuLaunchKernel(fn, grid, 1, 1, block, 1, 1, 0, nullptr, args, nullptr);
    if (c != CUDA_SUCCESS) die("cuLaunchKernel(copy)", c);
}

// ---------------------------------------------------------------------------
// Host reference: iterative radix-2 FFT in double precision.
// ---------------------------------------------------------------------------
void host_fft(std::vector<double>& re, std::vector<double>& im, bool inverse) {
    const size_t n = re.size();
    for (size_t i = 1, j = 0; i < n; ++i) {
        size_t bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) {
            std::swap(re[i], re[j]);
            std::swap(im[i], im[j]);
        }
    }
    const double pi = 3.14159265358979323846264338327950288;
    for (size_t len = 2; len <= n; len <<= 1) {
        const double ang = (inverse ? 2.0 : -2.0) * pi / (double)len;
        for (size_t i = 0; i < n; i += len) {
            for (size_t k = 0; k < len / 2; ++k) {
                const double wr = std::cos(ang * (double)k), wi = std::sin(ang * (double)k);
                const size_t a = i + k, b = i + k + len / 2;
                const double tr = re[b] * wr - im[b] * wi;
                const double ti = re[b] * wi + im[b] * wr;
                re[b] = re[a] - tr;
                im[b] = im[a] - ti;
                re[a] += tr;
                im[a] += ti;
            }
        }
    }
}

void write_file(const std::string& path, const void* data, size_t size) {
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) {
        std::fprintf(stderr, "cannot write %s\n", path.c_str());
        return;
    }
    std::fwrite(data, 1, size, f);
    std::fclose(f);
}

std::string plan_text(const VkFFTApplication& app) {
    std::string out;
    const VkFFTPlan* plan = app.localFFTPlan;
    const int uploads = (int)plan->numAxisUploads[0];
    char buf[256];
    for (int l = uploads - 1; l >= 0; --l) { // the order VkFFT launches them in
        const VkFFTAxis& ax = plan->axes[0][l];
        const VkFFTSpecializationConstantsLayout& sc = ax.specializationConstants;
        std::string radices;
        for (int s = 0; s < sc.numStages; ++s) {
            radices += (s ? "x" : "");
            radices += std::to_string(sc.stageRadix[s]);
        }
        std::snprintf(buf, sizeof buf,
                      "%s[len %lld radices %s reg/thr %d block %lldx%lld smem %lld B lut %d]",
                      out.empty() ? "" : " ", (long long)sc.fftDim.data.i, radices.c_str(),
                      sc.registers_per_thread, (long long)sc.localSize[0].data.i,
                      (long long)sc.localSize[1].data.i, (long long)sc.usedSharedMemory.data.i,
                      sc.LUT);
        out += buf;
    }
    return out;
}

Options parse(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&](const char* name) -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "%s needs a value\n", name);
                std::exit(2);
            }
            return argv[++i];
        };
        if (a == "--min-log2") o.min_log2 = std::atoi(next("--min-log2"));
        else if (a == "--max-log2") o.max_log2 = std::atoi(next("--max-log2"));
        else if (a == "--total-log2") o.total_log2 = std::atoi(next("--total-log2"));
        else if (a == "--rounds") o.rounds = std::atoi(next("--rounds"));
        else if (a == "--warmup") o.warmup = std::atoi(next("--warmup"));
        else if (a == "--queue-ms") o.queue_ms = std::atof(next("--queue-ms"));
        else if (a == "--forward-only") o.forward_only = true;
        else if (a == "--no-check") o.check = false;
        else if (a == "--csv") o.csv = next("--csv");
        else if (a == "--dump-dir") o.dump_dir = next("--dump-dir");
        else {
            std::fprintf(stderr,
                         "usage: vkfft_baseline [--min-log2 K] [--max-log2 K] [--total-log2 K]\n"
                         "                      [--rounds R] [--warmup W] [--queue-ms MS]\n"
                         "                      [--forward-only] [--no-check] [--csv FILE]\n"
                         "                      [--dump-dir DIR]\n");
            std::exit(2);
        }
    }
    if (o.max_log2 > o.total_log2) o.max_log2 = o.total_log2;
    if (o.warmup < 1) o.warmup = 1;
    return o;
}

} // namespace

int main(int argc, char** argv) {
    const Options opt = parse(argc, argv);

    CUresult c = cuInit(0);
    if (c != CUDA_SUCCESS) die("cuInit", c);
    if (cudaSetDevice(0) != cudaSuccess) die("cudaSetDevice", 0);
    CUdevice dev;
    c = cuDeviceGet(&dev, 0);
    if (c != CUDA_SUCCESS) die("cuDeviceGet", c);
    if (cudaDeviceSynchronize() != cudaSuccess) die("cudaDeviceSynchronize", 0);

    char name[256] = "";
    cuDeviceGetName(name, sizeof name, dev);
    int cc_major = 0, cc_minor = 0, smem_static = 0, smem_optin = 0, sm_count = 0;
    cuDeviceGetAttribute(&cc_major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev);
    cuDeviceGetAttribute(&cc_minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev);
    cuDeviceGetAttribute(&smem_static, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK, dev);
    cuDeviceGetAttribute(&smem_optin, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN, dev);
    cuDeviceGetAttribute(&sm_count, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev);

    const uint64_t total = 1ull << opt.total_log2; // complex elements
    const uint64_t bytes = total * 2ull * sizeof(float);
    const double gb = (double)bytes / 1e9;

    std::printf("device: %s  compute capability %d.%d  SMs %d\n", name, cc_major, cc_minor, sm_count);
    std::printf("shared memory per block: %d B static, %d B opt-in\n", smem_static, smem_optin);
    std::printf("VkFFT version %d   buffer: 2^%d complex fp32 = %.0f MB   rounds %d (+%d warm-up)\n\n",
                VkFFTGetVersion(), opt.total_log2, (double)bytes / (1024.0 * 1024.0), opt.rounds,
                opt.warmup);

    // Host data, once: uniform in [-1, 1).
    std::vector<float> host(total * 2);
    {
        uint64_t s = 0x9E3779B97F4A7C15ull;
        for (auto& x : host) {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            x = (float)((double)(s >> 11) / 9007199254740992.0 * 2.0 - 1.0);
        }
    }

    cuFloatComplex* buffer = nullptr;
    if (cudaMalloc((void**)&buffer, bytes) != cudaSuccess) die("cudaMalloc", 0);

    CopyKernels copy;
    if (!opt.forward_only) copy = build_copy_kernels(dev);

    FILE* csv = nullptr;
    if (!opt.csv.empty()) {
        csv = std::fopen(opt.csv.c_str(), "w");
        if (!csv) die("open csv", 0);
        std::fprintf(csv,
                     "log2n,n,batches,uploads,queue,fft_ms_med,fft_ms_min,fft_ms_p10,fft_ms_p90,"
                     "copy8_ms_med,copy8_ms_min,copy32_ms_med,copy32_ms_min,fwd1_ms_med,inv1_ms_med,"
                     "ns_per_fft,fft_over_copy,fft_GBps,copy_GBps,fwd_err,roundtrip_err,plan_ms,plan\n");
    }

    if (opt.forward_only)
        std::printf("%7s %9s %3s | %9s %9s\n", "n", "batches", "upl", "fwd1 ms", "min");
    else
        std::printf("%7s %9s %3s %4s | %8s %15s | %8s %8s | %8s %8s | %8s %6s %7s | %8s %8s\n", "n",
                    "batches", "upl", "q", "fft ms", "p10..p90", "copy8", "copy32", "fwd1", "inv1",
                    "ns/fft", "x copy", "GB/s", "fwd err", "rt err");

    for (int k = opt.min_log2; k <= opt.max_log2; ++k) {
        const uint64_t n = 1ull << k;
        const uint64_t batches = total / n;
        uint64_t buffer_size = bytes;

        VkFFTConfiguration cfg = {};
        VkFFTApplication app = {};
        cfg.FFTdim = 1;
        cfg.size[0] = n;
        cfg.numberBatches = batches;
        cfg.device = &dev;
        cfg.buffer = (void**)&buffer;
        cfg.bufferSize = &buffer_size;
        if (opt.forward_only) {
            cfg.makeForwardPlanOnly = 1;
        } else {
            cfg.normalize = 1;
        }
        const bool dump = !opt.dump_dir.empty();
        if (dump) {
            cfg.keepShaderCode = 1; // keeps the generated source; also prints it on every launch
            cfg.saveApplicationToString = 1;
        }

        const double t_plan0 = seconds_now();
        VkFFTResult res = initializeVkFFT(&app, cfg);
        const double plan_ms = (seconds_now() - t_plan0) * 1e3;
        if (res != VKFFT_SUCCESS) {
            std::printf("%7llu  initializeVkFFT failed: %s\n", (unsigned long long)n,
                        getVkFFTErrorString(res));
            continue;
        }
        const int uploads = (int)app.localFFTPlan->numAxisUploads[0];
        const std::string plan = plan_text(app);

        if (dump) {
            for (int l = 0; l < uploads; ++l) {
                const VkFFTAxis& ax = app.localFFTPlan->axes[0][l];
                const std::string stem = opt.dump_dir + "/vkfft_n" + std::to_string(n) + "_upload" +
                                         std::to_string(l);
                if (ax.specializationConstants.code0)
                    write_file(stem + ".cu", ax.specializationConstants.code0,
                               std::strlen(ax.specializationConstants.code0));
                if (ax.binary) write_file(stem + ".cubin", ax.binary, (size_t)ax.binarySize);
            }
            std::printf("%7llu  dumped %d upload(s): %s\n", (unsigned long long)n, uploads,
                        plan.c_str());
            deleteVkFFT(&app);
            continue; // a dump run does not time: keepShaderCode prints inside the launch
        }

        VkFFTLaunchParams lp = {};
        auto sync = [] {
            if (cudaDeviceSynchronize() != cudaSuccess) die("cudaDeviceSynchronize", 0);
        };
        auto upload = [&] {
            if (cudaMemcpy(buffer, host.data(), bytes, cudaMemcpyHostToDevice) != cudaSuccess)
                die("cudaMemcpy to device", 0);
        };
        auto transform = [&](int inverse) {
            const VkFFTResult r = VkFFTAppend(&app, inverse, &lp);
            if (r != VKFFT_SUCCESS) die("VkFFTAppend", r);
        };

        // ---- correctness: first and last sequence of the batch ----
        double fwd_err = -1.0, rt_err = -1.0;
        if (opt.check && !opt.forward_only) {
            upload();
            transform(-1);
            sync();
            std::vector<float> got(2 * n);
            fwd_err = 0.0;
            for (uint64_t which : {uint64_t(0), batches - 1}) {
                const uint64_t off = which * n;
                if (cudaMemcpy(got.data(), buffer + off, 2 * n * sizeof(float),
                               cudaMemcpyDeviceToHost) != cudaSuccess)
                    die("cudaMemcpy to host", 0);
                std::vector<double> re(n), im(n);
                for (uint64_t i = 0; i < n; ++i) {
                    re[i] = host[2 * (off + i)];
                    im[i] = host[2 * (off + i) + 1];
                }
                host_fft(re, im, false);
                double num = 0.0, den = 0.0;
                for (uint64_t i = 0; i < n; ++i) {
                    const double dr = got[2 * i] - re[i], di = got[2 * i + 1] - im[i];
                    num += dr * dr + di * di;
                    den += re[i] * re[i] + im[i] * im[i];
                }
                fwd_err = std::max(fwd_err, std::sqrt(num / den));
            }
            transform(1);
            sync();
            rt_err = 0.0;
            for (uint64_t which : {uint64_t(0), batches - 1}) {
                const uint64_t off = which * n;
                if (cudaMemcpy(got.data(), buffer + off, 2 * n * sizeof(float),
                               cudaMemcpyDeviceToHost) != cudaSuccess)
                    die("cudaMemcpy to host", 0);
                double num = 0.0, den = 0.0;
                for (uint64_t i = 0; i < 2 * n; ++i) {
                    const double d = (double)got[i] - (double)host[2 * off + i];
                    num += d * d;
                    den += (double)host[2 * off + i] * (double)host[2 * off + i];
                }
                rt_err = std::max(rt_err, std::sqrt(num / den));
            }
        }

        // ---- timing ----
        upload();
        sync();
        const unsigned block = 256;
        const unsigned grid8 = (unsigned)((total + block - 1) / block);
        const unsigned grid32 = (unsigned)((total / 4 + block - 1) / block);
        unsigned long long n8 = total;      // complex elements
        unsigned long long n16 = total / 2; // 16-byte records, two per thread
        void* buf_ptr = buffer;
        void* args8[] = {&buf_ptr, &n8};
        void* args32[] = {&buf_ptr, &n16};

        if (opt.forward_only) {
            std::vector<double> t_fwd;
            for (int r = 0; r < opt.warmup + opt.rounds; ++r) {
                const double t0 = seconds_now();
                transform(-1);
                sync();
                if (r >= opt.warmup) t_fwd.push_back((seconds_now() - t0) * 1e3);
            }
            const Stat f = stat_of(t_fwd);
            std::printf("%7llu %9llu %3d | %9.4f %9.4f\n", (unsigned long long)n,
                        (unsigned long long)batches, uploads, f.med, f.min);
            std::printf("        plan: %s\n", plan.c_str());
            deleteVkFFT(&app);
            continue;
        }

        // Warm-up, and the queue length: enough forward + inverse pairs for about queue_ms.
        double pair_ms = 0.0;
        for (int r = 0; r < opt.warmup; ++r) {
            const double t0 = seconds_now();
            for (int q = 0; q < 4; ++q) {
                transform(-1);
                transform(1);
            }
            sync();
            pair_ms = (seconds_now() - t0) * 1e3 / 4.0;
            launch(copy.copy8, grid8, block, args8);
            launch(copy.copy8, grid8, block, args8);
            launch(copy.copy32, grid32, block, args32);
            launch(copy.copy32, grid32, block, args32);
            sync();
        }
        const int pairs = (int)std::min(512.0, std::max(2.0, std::round(opt.queue_ms / pair_ms)));
        const int queue = 2 * pairs; // launches in a queued sample; even, so the copies undo themselves

        std::vector<double> t_fft, t_c8, t_c32, t_fwd1, t_inv1;
        for (int r = 0; r < opt.rounds; ++r) {
            double t0 = seconds_now();
            for (int q = 0; q < pairs; ++q) {
                transform(-1);
                transform(1);
            }
            sync();
            t_fft.push_back((seconds_now() - t0) * 1e3 / (double)queue);

            t0 = seconds_now();
            for (int q = 0; q < queue; ++q) launch(copy.copy8, grid8, block, args8);
            sync();
            t_c8.push_back((seconds_now() - t0) * 1e3 / (double)queue);

            t0 = seconds_now();
            for (int q = 0; q < queue; ++q) launch(copy.copy32, grid32, block, args32);
            sync();
            t_c32.push_back((seconds_now() - t0) * 1e3 / (double)queue);

            t0 = seconds_now();
            transform(-1);
            sync();
            double t1 = seconds_now();
            t_fwd1.push_back((t1 - t0) * 1e3);
            transform(1);
            sync();
            t_inv1.push_back((seconds_now() - t1) * 1e3);
        }

        const Stat f = stat_of(t_fft), c8 = stat_of(t_c8), c32 = stat_of(t_c32);
        const Stat f1 = stat_of(t_fwd1), i1 = stat_of(t_inv1);
        const double copy_med = std::min(c8.med, c32.med);
        const double ns_per_fft = f.med * 1e6 / (double)batches;
        const double fft_gbps = 2.0 * gb / (f.med * 1e-3);
        const double copy_gbps = 2.0 * gb / (copy_med * 1e-3);
        const double over = f.med / copy_med;

        std::printf("%7llu %9llu %3d %4d | %8.4f %7.4f..%-7.4f | %8.4f %8.4f | %8.4f %8.4f | %8.1f %6.2f %7.0f | %8.1e %8.1e\n",
                    (unsigned long long)n, (unsigned long long)batches, uploads, queue, f.med, f.p10,
                    f.p90, c8.med, c32.med, f1.med, i1.med, ns_per_fft, over, fft_gbps, fwd_err,
                    rt_err);
        std::printf("        plan: %s  (planned in %.0f ms)\n", plan.c_str(), plan_ms);
        if (csv) {
            std::fprintf(csv,
                         "%d,%llu,%llu,%d,%d,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.2f,"
                         "%.3f,%.1f,%.1f,%.3e,%.3e,%.0f,\"%s\"\n",
                         k, (unsigned long long)n, (unsigned long long)batches, uploads, queue, f.med,
                         f.min, f.p10, f.p90, c8.med, c8.min, c32.med, c32.min, f1.med, i1.med,
                         ns_per_fft, over, fft_gbps, copy_gbps, fwd_err, rt_err, plan_ms,
                         plan.c_str());
            std::fflush(csv);
        }
        deleteVkFFT(&app);
    }

    if (csv) std::fclose(csv);
    cudaFree(buffer);
    return 0;
}
