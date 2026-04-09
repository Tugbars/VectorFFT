/**
 * spectrum_analyzer.c — Live audio spectrum analyzer with VectorFFT
 *
 * Captures audio from the default microphone (or system loopback) via
 * miniaudio, runs R2C FFT with a Hann window, and displays the spectrum
 * as a live-updating ASCII bar chart in the terminal.
 *
 * Modes:
 *   --mic       Capture from microphone (default)
 *   --loopback  Capture system audio (what's playing — WASAPI only)
 *   --test      No audio capture, use synthetic test signal
 *
 * Build:
 *   cmake --build build --target vfft_spectrum
 *
 * Requires: miniaudio.h in the same directory (single-header, MIT license).
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define MINIAUDIO_IMPLEMENTATION
#include "miniaudio.h"

#include "vfft.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ═══════════════════════════════════════════════════════════════
 * CONFIG
 * ═══════════════════════════════════════════════════════════════ */

#define FFT_N          4096        /* FFT length (4K ≈ 93ms at 44.1k, 10.7 Hz/bin) */
#define SAMPLE_RATE    44100
#define NUM_BARS       32          /* frequency bands in the display */
#define DB_FLOOR       -90.0
#define DB_CEIL        0.0
#define FPS            30
#define K_BATCH        4           /* min SIMD width; channel 0 used */


/* ═══════════════════════════════════════════════════════════════
 * RING BUFFER — audio thread writes, main thread reads
 *
 * Lock-free single-producer single-consumer via atomic write index.
 * The ring is oversized (4x FFT length) so the reader never catches
 * the writer even at high latency.
 * ═══════════════════════════════════════════════════════════════ */

#define RING_SIZE (FFT_N * 4)
#define RING_MASK (RING_SIZE - 1)   /* power-of-2 assumed */

static float g_ring[RING_SIZE];
static volatile long g_write_pos = 0;
static long g_read_pos = 0;


/* ═══════════════════════════════════════════════════════════════
 * AUDIO CALLBACK — called by miniaudio on the audio thread
 * ═══════════════════════════════════════════════════════════════ */

static void audio_callback(ma_device *dev, void *output,
                           const void *input, ma_uint32 frame_count) {
    (void)dev; (void)output;
    const float *pcm = (const float *)input;

    for (ma_uint32 i = 0; i < frame_count; i++) {
        g_ring[g_write_pos & RING_MASK] = pcm[i];
        g_write_pos++;
    }
}


/* ═══════════════════════════════════════════════════════════════
 * WINDOW + SPECTRUM
 * ═══════════════════════════════════════════════════════════════ */

static void window_hann(double *w, int N) {
    for (int n = 0; n < N; n++)
        w[n] = 0.5 * (1.0 - cos(2.0 * M_PI * n / (N - 1)));
}

static void compute_spectrum(
    vfft_plan plan, const double *signal, const double *window,
    double *work, double *out_re, double *out_im,
    double *spectrum_dB, int N, int K)
{
    memset(work, 0, (size_t)N * K * sizeof(double));
    for (int n = 0; n < N; n++)
        work[n * K] = signal[n] * window[n];

    vfft_execute_r2c(plan, work, out_re, out_im);

    double ref = (double)N / 2.0;
    for (int k = 0; k <= N / 2; k++) {
        double re = out_re[k * K];
        double im = out_im[k * K];
        double mag = sqrt(re * re + im * im);
        spectrum_dB[k] = 20.0 * log10(mag / ref + 1e-300);
    }
}


/* ═══════════════════════════════════════════════════════════════
 * ASCII RENDERER — clears screen and redraws each frame
 * ═══════════════════════════════════════════════════════════════ */

static void render_spectrum(const double *spectrum_dB, int N,
                            double sample_rate) {
    double nyquist = sample_rate / 2.0;
    double log_lo = log10(20.0);
    double log_hi = log10(nyquist);
    int bar_width = 60;

    /* Move cursor to top-left (ANSI escape) */
    printf("\033[H");

    printf("  VectorFFT Live Spectrum  |  N=%d  fs=%d  %d Hz/bin\n",
           N, (int)sample_rate, (int)(sample_rate / N));
    printf("  %-8s %-6s ", "Freq", "dB");
    for (int i = 0; i < bar_width; i++) printf("-");
    printf("\n");

    for (int b = 0; b < NUM_BARS; b++) {
        double f_lo = pow(10.0, log_lo + (log_hi - log_lo) * b / NUM_BARS);
        double f_hi = pow(10.0, log_lo + (log_hi - log_lo) * (b + 1) / NUM_BARS);
        double f_center = sqrt(f_lo * f_hi);

        int k_lo = (int)(f_lo * N / sample_rate);
        int k_hi = (int)(f_hi * N / sample_rate);
        if (k_lo < 0) k_lo = 0;
        if (k_hi > N / 2) k_hi = N / 2;
        if (k_lo >= k_hi) k_lo = k_hi > 0 ? k_hi - 1 : 0;

        double peak = -999.0;
        for (int k = k_lo; k <= k_hi; k++)
            if (spectrum_dB[k] > peak) peak = spectrum_dB[k];

        int filled = (int)((peak - DB_FLOOR) / (DB_CEIL - DB_FLOOR) * bar_width);
        if (filled < 0) filled = 0;
        if (filled > bar_width) filled = bar_width;

        if (f_center >= 1000.0)
            printf("  %5.1fk ", f_center / 1000.0);
        else
            printf("  %5.0f  ", f_center);

        printf("%+6.1f |", peak);

        /* Color: green for low, yellow for mid, red for loud */
        for (int i = 0; i < filled; i++) {
            if (i < bar_width * 6 / 10)
                printf("\033[32m#\033[0m");      /* green */
            else if (i < bar_width * 85 / 100)
                printf("\033[33m#\033[0m");      /* yellow */
            else
                printf("\033[31m#\033[0m");      /* red */
        }
        for (int i = filled; i < bar_width; i++) printf(" ");
        printf("|\n");
    }

    printf("\n  Press Ctrl+C to exit.\n");
    fflush(stdout);
}


/* ═══════════════════════════════════════════════════════════════
 * TEST SIGNAL (for --test mode)
 * ═══════════════════════════════════════════════════════════════ */

static void generate_test_signal(double *x, int N, double sr) {
    for (int n = 0; n < N; n++) {
        double t = (double)n / sr;
        x[n]  = 1.0   * sin(2*M_PI*1000*t);
        x[n] += 0.1   * sin(2*M_PI*2000*t);
        x[n] += 0.01  * sin(2*M_PI*3000*t);
        x[n] += 0.001 * sin(2*M_PI*5000*t);
        x[n] += 0.0001 * ((double)rand()/RAND_MAX*2-1);
    }
}


/* ═══════════════════════════════════════════════════════════════
 * MAIN
 * ═══════════════════════════════════════════════════════════════ */

#ifdef _WIN32
#include <windows.h>
#define sleep_ms(ms) Sleep(ms)
#else
#include <unistd.h>
#define sleep_ms(ms) usleep((ms)*1000)
#endif

int main(int argc, char **argv) {
    int use_loopback = 0;
    int use_test = 0;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--loopback") == 0) use_loopback = 1;
        if (strcmp(argv[i], "--test") == 0)     use_test = 1;
    }

    /* Init VectorFFT */
    vfft_init();
    vfft_plan plan = vfft_plan_r2c(FFT_N, K_BATCH);
    if (!plan) {
        fprintf(stderr, "Failed to create R2C plan for N=%d\n", FFT_N);
        return 1;
    }

    /* Allocate buffers */
    const int N = FFT_N, K = K_BATCH;
    double *signal      = vfft_alloc(N * sizeof(double));
    double *work        = vfft_alloc(N * K * sizeof(double));
    double *window      = vfft_alloc(N * sizeof(double));
    double *out_re      = vfft_alloc(N * K * sizeof(double));
    double *out_im      = vfft_alloc(N * K * sizeof(double));
    double *spectrum_dB = (double *)malloc((N / 2 + 1) * sizeof(double));

    window_hann(window, N);

    /* Init audio (unless --test mode) */
    ma_device device;
    int audio_active = 0;

    if (!use_test) {
        ma_device_config cfg;
        if (use_loopback) {
            cfg = ma_device_config_init(ma_device_type_loopback);
            printf("Mode: system audio loopback (WASAPI)\n");
        } else {
            cfg = ma_device_config_init(ma_device_type_capture);
            printf("Mode: microphone capture\n");
        }
        cfg.capture.format   = ma_format_f32;
        cfg.capture.channels = 1;
        cfg.sampleRate       = SAMPLE_RATE;
        cfg.dataCallback     = audio_callback;

        if (ma_device_init(NULL, &cfg, &device) != MA_SUCCESS) {
            fprintf(stderr, "Failed to init audio device.\n");
            fprintf(stderr, "Try --test mode for synthetic signal.\n");
            return 1;
        }
        ma_device_start(&device);
        audio_active = 1;
        printf("Audio device: %s\n", device.capture.name);
    } else {
        printf("Mode: synthetic test signal (1 kHz + harmonics)\n");
    }

    printf("FFT: N=%d, %.1f Hz/bin, %d bands\n\n", N,
           (double)SAMPLE_RATE / N, NUM_BARS);

    /* Clear screen (ANSI) */
    printf("\033[2J");

    /* ── Main loop ── */
    for (;;) {
        if (use_test) {
            generate_test_signal(signal, N, SAMPLE_RATE);
        } else {
            /* Read N samples from ring buffer */
            long wp = g_write_pos;
            long available = wp - g_read_pos;
            if (available < N) {
                sleep_ms(1000 / FPS);
                continue;  /* wait for enough data */
            }
            /* Take the most recent N samples */
            g_read_pos = wp - N;
            for (int n = 0; n < N; n++)
                signal[n] = (double)g_ring[(g_read_pos + n) & RING_MASK];
            g_read_pos = wp;
        }

        compute_spectrum(plan, signal, window, work, out_re, out_im,
                         spectrum_dB, N, K);
        render_spectrum(spectrum_dB, N, SAMPLE_RATE);

        sleep_ms(1000 / FPS);
    }

    /* Cleanup (unreachable, but good form) */
    if (audio_active)
        ma_device_uninit(&device);
    vfft_destroy(plan);
    vfft_free(signal); vfft_free(work); vfft_free(window);
    vfft_free(out_re); vfft_free(out_im);
    free(spectrum_dB);
    return 0;
}
