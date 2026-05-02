/**
 * stride_workspace.h -- Pre-allocated memory pool for FFT plans
 *
 * Reduces malloc/free overhead when creating many plans (e.g., during
 * exhaustive search, or when cycling through different FFT sizes).
 *
 * Design:
 *   - Simple bump allocator with 64-byte aligned returns
 *   - Falls back to heap allocation when pool is exhausted
 *   - Reset releases all pool allocations at once (no individual free)
 *   - Heap fallback allocations are tracked and freed individually
 *
 * NOT wired into the executor yet. This is the API and implementation
 * ready for integration when plan creation is updated to accept a
 * workspace parameter.
 *
 * Usage:
 *   stride_workspace_t ws;
 *   stride_workspace_init(&ws, 4 * 1024 * 1024);  // 4 MB pool
 *
 *   // Allocate from pool (64-byte aligned)
 *   double *buf = stride_ws_alloc(&ws, 1024 * sizeof(double));
 *
 *   // Reset pool (all pool allocations become invalid)
 *   stride_workspace_reset(&ws);
 *
 *   // Destroy pool (frees underlying memory)
 *   stride_workspace_destroy(&ws);
 *
 * Future integration:
 *   stride_plan_create_ws(..., &ws);  // plan allocates from workspace
 *   stride_workspace_reset(&ws);      // after plan is destroyed
 */
#ifndef STRIDE_WORKSPACE_H
#define STRIDE_WORKSPACE_H

#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#  include <malloc.h>
#endif

#define STRIDE_WS_ALIGN 64
#define STRIDE_WS_MAX_FALLBACKS 256

typedef struct {
    /* Pool memory (single contiguous block, 64-byte aligned) */
    char  *pool;          /* base pointer (for free) */
    size_t pool_size;     /* total pool size in bytes */
    size_t pool_used;     /* current bump offset */

    /* Heap fallback tracking (when pool is exhausted) */
    void  *fallbacks[STRIDE_WS_MAX_FALLBACKS];
    int    n_fallbacks;

    /* Statistics */
    size_t peak_used;     /* high-water mark within pool */
    size_t total_allocated;  /* total bytes requested (pool + fallback) */
    int    n_pool_allocs;    /* number of pool allocations */
    int    n_fallback_allocs; /* number of heap fallback allocations */
} stride_workspace_t;

/**
 * stride_workspace_init -- Create a workspace with the given pool size.
 *
 * pool_bytes: size of the pre-allocated pool. Larger = fewer fallbacks
 *             to heap. 0 = no pool (all allocations go to heap).
 *
 * Guideline: for an N-point FFT with batch K, the plan needs roughly
 *   twiddles: sum over stages of (R_s - 1) * K * 16 bytes
 *   scratch (Bluestein/Rader): M * B * 16 bytes
 * A 4 MB pool covers most cases up to N=100000.
 */
static int stride_workspace_init(stride_workspace_t *ws, size_t pool_bytes) {
    memset(ws, 0, sizeof(*ws));

    if (pool_bytes > 0) {
        /* Round up to alignment */
        pool_bytes = (pool_bytes + STRIDE_WS_ALIGN - 1) & ~(size_t)(STRIDE_WS_ALIGN - 1);

#ifdef _WIN32
        ws->pool = (char *)_aligned_malloc(pool_bytes, STRIDE_WS_ALIGN);
#else
        ws->pool = NULL;
        if (posix_memalign((void **)&ws->pool, STRIDE_WS_ALIGN, pool_bytes) != 0)
            ws->pool = NULL;
#endif
        if (!ws->pool) return -1;
        ws->pool_size = pool_bytes;
    }
    return 0;
}

/**
 * stride_ws_alloc -- Allocate bytes from the workspace.
 *
 * Returns a 64-byte aligned pointer. Falls back to heap if pool
 * is exhausted. Returns NULL only on total allocation failure.
 */
static void *stride_ws_alloc(stride_workspace_t *ws, size_t bytes) {
    if (bytes == 0) return NULL;

    /* Round up to 64-byte alignment */
    size_t aligned_bytes = (bytes + STRIDE_WS_ALIGN - 1) & ~(size_t)(STRIDE_WS_ALIGN - 1);

    ws->total_allocated += bytes;

    /* Try pool first */
    if (ws->pool && ws->pool_used + aligned_bytes <= ws->pool_size) {
        void *p = ws->pool + ws->pool_used;
        ws->pool_used += aligned_bytes;
        ws->n_pool_allocs++;
        if (ws->pool_used > ws->peak_used)
            ws->peak_used = ws->pool_used;
        return p;
    }

    /* Fallback to heap */
    if (ws->n_fallbacks >= STRIDE_WS_MAX_FALLBACKS) return NULL;

    void *p;
#ifdef _WIN32
    p = _aligned_malloc(aligned_bytes, STRIDE_WS_ALIGN);
#else
    if (posix_memalign(&p, STRIDE_WS_ALIGN, aligned_bytes) != 0)
        p = NULL;
#endif
    if (p) {
        ws->fallbacks[ws->n_fallbacks++] = p;
        ws->n_fallback_allocs++;
    }
    return p;
}

/**
 * stride_workspace_reset -- Release all allocations from the workspace.
 *
 * Pool allocations are released by resetting the bump pointer (instant).
 * Heap fallback allocations are individually freed.
 *
 * After reset, all previously returned pointers are invalid.
 */
static void stride_workspace_reset(stride_workspace_t *ws) {
    /* Reset pool bump pointer */
    ws->pool_used = 0;

    /* Free heap fallbacks */
    for (int i = 0; i < ws->n_fallbacks; i++) {
#ifdef _WIN32
        _aligned_free(ws->fallbacks[i]);
#else
        free(ws->fallbacks[i]);
#endif
    }
    ws->n_fallbacks = 0;

    /* Keep statistics (don't reset peak/total — useful for sizing) */
}

/**
 * stride_workspace_destroy -- Free the workspace and all its memory.
 */
static void stride_workspace_destroy(stride_workspace_t *ws) {
    stride_workspace_reset(ws);
    if (ws->pool) {
#ifdef _WIN32
        _aligned_free(ws->pool);
#else
        free(ws->pool);
#endif
    }
    memset(ws, 0, sizeof(*ws));
}

/**
 * stride_workspace_stats -- Print workspace usage statistics.
 *
 * Useful for tuning pool size: if n_fallback_allocs > 0, the pool
 * was too small and some allocations went to the heap.
 */
static void stride_workspace_stats(const stride_workspace_t *ws) {
    printf("Workspace: pool=%zuKB used=%zuKB peak=%zuKB | "
           "allocs: %d pool, %d fallback | total requested: %zuKB\n",
           ws->pool_size / 1024,
           ws->pool_used / 1024,
           ws->peak_used / 1024,
           ws->n_pool_allocs,
           ws->n_fallback_allocs,
           ws->total_allocated / 1024);
}

#endif /* STRIDE_WORKSPACE_H */
