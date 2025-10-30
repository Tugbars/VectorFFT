/**
 * @file fft_transpose.c
 * @brief Cache-oblivious in-place transpose implementation
 * 
 * @details
 * This file implements the cache-oblivious recursive transpose algorithm
 * for VectorFFT's four-step FFT decomposition.
 * 
 * **Key Implementation Details:**
 * 
 * 1. **Tile Size Calculation:**
 *    - Computed once based on L1 cache size
 *    - Formula: tile_size = sqrt(L1_cache / (2 × element_size))
 *    - Factor of 2 because we need TWO tiles in cache simultaneously for swapping
 * 
 * 2. **Recursion Structure:**
 *    - Base case: n ≤ tile_size → transpose directly
 *    - Recursive case: Divide into quadrants, swap off-diagonal, recurse on diagonal
 * 
 * 3. **Diagonal vs Off-Diagonal:**
 *    - Diagonal blocks (A, D): Must transpose in-place (can't swap with self!)
 *    - Off-diagonal blocks (B, C): Swap with transpose partner
 * 
 * 4. **Memory Layout:**
 *    - Row-major: element[i,j] at index i*stride + j
 *    - Transpose: swap element[i,j] ↔ element[j,i]
 * 
 * **Algorithm Complexity:**
 * - Time: O(n²) element swaps (optimal)
 * - Space: O(1) extra memory, O(log n) stack (optimal)
 * - Cache misses: O(n²/B + n²/√ZB) where B=line size, Z=cache size (optimal!)
 * 
 * @author VectorFFT Team
 * @date 2025
 * @version 1.0
 */

#include "fft_transpose.h"
#include <assert.h>

//==============================================================================
// INTERNAL HELPER: TILE SIZE COMPUTATION
//==============================================================================

/**
 * @brief Compute optimal tile size for cache-efficient transpose
 * 
 * @details
 * The tile size is chosen so that TWO tiles fit comfortably in L1 cache.
 * We need two tiles because we're swapping pairs of tiles during transpose.
 * 
 * **Derivation:**
 * @code
 * Let:
 *   L1 = L1 cache size (bytes)
 *   E  = element size (bytes)
 *   T  = tile dimension (elements per side)
 * 
 * Requirement: 2 tiles must fit in L1
 *   2 × (T × T × E) ≤ L1
 *   T² ≤ L1 / (2E)
 *   T = sqrt(L1 / (2E))
 * @endcode
 * 
 * **Example (32 KB L1, complex double):**
 * @code
 * element_size = 16 bytes (2 × double)
 * tile_size = sqrt(32768 / (2 × 16))
 *           = sqrt(32768 / 32)
 *           = sqrt(1024)
 *           = 32 elements
 * 
 * Verification:
 *   2 tiles = 2 × (32 × 32 × 16) = 32768 bytes = 32 KB ✓
 * @endcode
 * 
 * @param[in] element_size Size of one matrix element in bytes
 * @return Optimal tile dimension (elements per side)
 * 
 * @note Uses integer square root for portability (no libm dependency)
 */
static inline size_t compute_tile_size(size_t element_size)
{
    // Reserve space for TWO tiles in cache
    size_t cache_for_two_tiles = FFT_TRANSPOSE_L1_CACHE_SIZE / 2;
    size_t elements_per_tile = cache_for_two_tiles / element_size;
    
    // Tile is square: tile_size × tile_size = elements_per_tile
    // Use integer square root (no floating point needed)
    size_t tile_size = 1;
    while (tile_size * tile_size < elements_per_tile) {
        tile_size++;
    }
    
    // Ensure tile_size is at least 8 (avoid excessive recursion for tiny caches)
    return (tile_size < 8) ? 8 : tile_size;
}

//==============================================================================
// LEVEL 1: TILE SWAPPING (Base Case Operations)
//==============================================================================

/**
 * @brief Swap two rectangular tiles in a complex matrix
 * 
 * @details
 * This is the innermost hot loop of the transpose algorithm. It swaps a tile
 * at position [r1, c1] with its transpose partner at position [r2, c2].
 * 
 * **Memory Access Pattern:**
 * @code
 * For tile1 at [r1, c1] and tile2 at [r2, c2]:
 *   Swap element[r1+i, c1+j] ↔ element[r2+j, c2+i] for all i,j in tile
 * @endcode
 * 
 * Note the index swap: (i,j) ↔ (j,i) implements the transpose.
 * 
 * **Cache Behavior:**
 * - Both tiles should fit in L1 (by design of tile_size)
 * - First access to each tile: cache miss (unavoidable)
 * - All subsequent accesses: cache hit (tile stays hot)
 * 
 * @param[in,out] mat Pointer to matrix data
 * @param[in] stride Number of columns in full matrix (for linear indexing)
 * @param[in] r1 Starting row of first tile
 * @param[in] r1_end Ending row of first tile (exclusive)
 * @param[in] c1 Starting column of first tile
 * @param[in] c1_end Ending column of first tile (exclusive)
 * @param[in] r2 Starting row of second tile
 * @param[in] c2 Starting column of second tile
 * 
 * @note Tile dimensions must match: (r1_end-r1) × (c1_end-c1) = tile size
 * @note This function does not check bounds - caller must ensure validity
 */
static void swap_tile_complex(
    fft_complex *mat,
    size_t stride,
    size_t r1, size_t r1_end,
    size_t c1, size_t c1_end,
    size_t r2, size_t c2)
{
    const size_t tile_rows = r1_end - r1;
    const size_t tile_cols = c1_end - c1;
    
    // Swap each element in the tile with its transpose partner
    for (size_t i = 0; i < tile_rows; i++) {
        for (size_t j = 0; j < tile_cols; j++) {
            // Linear indices in row-major layout
            size_t idx1 = (r1 + i) * stride + (c1 + j);
            size_t idx2 = (r2 + j) * stride + (c2 + i);  // Note: j and i swapped!
            
            // Three-way swap using temporary
            fft_complex temp = mat[idx1];
            mat[idx1] = mat[idx2];
            mat[idx2] = temp;
        }
    }
}

/**
 * @brief Transpose a diagonal tile in-place
 * 
 * @details
 * For diagonal tiles (where row_offset == col_offset), we can't swap with
 * another tile. Instead, we transpose the tile in-place by swapping elements
 * above the diagonal with elements below the diagonal.
 * 
 * **Algorithm:**
 * @code
 * For diagonal tile starting at [r, r]:
 *   for i in 0..dim:
 *     for j in i+1..dim:
 *       swap element[r+i, r+j] ↔ element[r+j, r+i]
 * @endcode
 * 
 * This is the standard in-place transpose for a small square matrix.
 * 
 * @param[in,out] mat Pointer to matrix data
 * @param[in] stride Number of columns in full matrix
 * @param[in] row_off Starting row of diagonal tile
 * @param[in] dim Tile dimension (number of rows/cols)
 * 
 * @note Only processes upper triangle; lower triangle is mirror image
 */
static void transpose_diagonal_tile(
    fft_complex *mat,
    size_t stride,
    size_t row_off,
    size_t dim)
{
    // Transpose in-place: swap upper and lower triangles
    for (size_t i = 0; i < dim; i++) {
        for (size_t j = i + 1; j < dim; j++) {
            size_t idx1 = (row_off + i) * stride + (row_off + j);
            size_t idx2 = (row_off + j) * stride + (row_off + i);
            
            fft_complex temp = mat[idx1];
            mat[idx1] = mat[idx2];
            mat[idx2] = temp;
        }
    }
}

//==============================================================================
// LEVEL 2: TILED RECTANGLE PROCESSING
//==============================================================================

/**
 * @brief Process an off-diagonal rectangle using cache-sized tiles
 * 
 * @details
 * Divide a large rectangle into tile_size × tile_size blocks and process
 * each block independently. This ensures all accesses stay in L1 cache.
 * 
 * **Example:** For a 128×64 rectangle with tile_size=32:
 * @code
 *   ┌────┬────┬────┬────┐
 *   │ T1 │ T2 │ T3 │ T4 │  Each tile: 32×32
 *   ├────┼────┼────┼────┤  Process one at a time
 *   │ T5 │ T6 │ T7 │ T8 │  Total: 8 tiles (4×2)
 *   └────┴────┴────┴────┘
 * @endcode
 * 
 * **Cache Behavior:**
 * - Each tile pair: one load (miss), N² swaps (all hits)
 * - Total: ~8 cache misses instead of millions for naive approach
 * 
 * @param[in,out] mat Pointer to matrix data
 * @param[in] stride Number of columns in full matrix
 * @param[in] r_start Starting row of rectangle
 * @param[in] r_end Ending row of rectangle (exclusive)
 * @param[in] c_start Starting column of rectangle
 * @param[in] c_end Ending column of rectangle (exclusive)
 * @param[in] tile_sz Tile dimension for blocking
 * 
 * @note Handles partial tiles at boundaries automatically
 */
static void process_rectangle_tiled(
    fft_complex *mat,
    size_t stride,
    size_t r_start, size_t r_end,
    size_t c_start, size_t c_end,
    size_t tile_sz)
{
    // Iterate over tiles in the rectangle
    for (size_t tile_row = r_start; tile_row < r_end; tile_row += tile_sz) {
        for (size_t tile_col = c_start; tile_col < c_end; tile_col += tile_sz) {
            
            // Calculate tile bounds (handle partial tiles at edges)
            size_t tile_row_end = tile_row + tile_sz;
            if (tile_row_end > r_end) {
                tile_row_end = r_end;
            }
            
            size_t tile_col_end = tile_col + tile_sz;
            if (tile_col_end > c_end) {
                tile_col_end = c_end;
            }
            
            // Swap this tile with its transpose partner
            // For transpose: position [i,j] swaps with [j,i]
            swap_tile_complex(mat, stride,
                            tile_row, tile_row_end,
                            tile_col, tile_col_end,
                            tile_col, tile_row);
        }
    }
}

//==============================================================================
// LEVEL 3: RECURSIVE TRANSPOSE (Core Algorithm)
//==============================================================================

/**
 * @brief Recursive cache-oblivious transpose implementation
 * 
 * @details
 * This is the heart of the algorithm. It recursively divides the matrix
 * into quadrants until reaching a base case that fits in L1 cache.
 * 
 * **Algorithm Overview:**
 * @code
 * transpose_recursive(matrix, n, row_offset, col_offset):
 *   if n ≤ tile_size:
 *     // BASE CASE: Small enough to fit in L1
 *     if row_offset == col_offset:
 *       transpose_diagonal_tile()    // Can't swap with self!
 *     else:
 *       swap_tile_with_partner()     // Swap with transpose location
 *   else:
 *     // RECURSIVE CASE: Divide and conquer
 *     Divide into 4 quadrants:
 *       ┌─────┬─────┐
 *       │  A  │  B  │
 *       ├─────┼─────┤
 *       │  C  │  D  │
 *       └─────┴─────┘
 *     
 *     Step 1: Swap B ↔ C (in tiles)
 *     Step 2: Recurse on A (transpose top-left)
 *     Step 3: Recurse on D (transpose bottom-right)
 * @endcode
 * 
 * **Why This Works:**
 * 
 * Transpose of full matrix:
 * @code
 *   [ A  B ]ᵀ   [ Aᵀ  Cᵀ ]
 *   [ C  D ]  = [ Bᵀ  Dᵀ ]
 * @endcode
 * 
 * After swapping B↔C, we have:
 * @code
 *   [ A  C ]
 *   [ B  D ]
 * @endcode
 * 
 * Then recursively transpose A and D:
 * @code
 *   [ Aᵀ  C  ]
 *   [ B   Dᵀ ]
 * @endcode
 * 
 * But wait! B and C are now in the right positions to BE transposed
 * by the recursive calls. When we swapped B↔C, we effectively transposed
 * them as well (by definition of the swap operation).
 * 
 * Final result:
 * @code
 *   [ Aᵀ  Cᵀ ]
 *   [ Bᵀ  Dᵀ ]  ← Correct transpose!
 * @endcode
 * 
 * **Recursion Tree Example (n=1024):**
 * @code
 * Level 0: n=1024 (1 MB)     ← Too big for any cache
 *   ├─ Level 1: n=512 (256 KB)  ← Too big for L2
 *   │  ├─ Level 2: n=256 (64 KB)   ← Fits in L3
 *   │  │  ├─ Level 3: n=128 (16 KB)  ← Fits in L2
 *   │  │  │  ├─ Level 4: n=64 (4 KB)   ← Fits in L1
 *   │  │  │  │  ├─ Level 5: n=32 (1 KB)  ← BASE CASE ✓
 * @endcode
 * 
 * At each level, working set fits in SOME cache, ensuring good performance
 * at all scales. This is the "cache-oblivious" property.
 * 
 * @param[in,out] mat Pointer to matrix data
 * @param[in] dim Dimension of current submatrix (n for n×n)
 * @param[in] row_off Row offset in full matrix (for submatrix addressing)
 * @param[in] col_off Column offset in full matrix
 * @param[in] stride Total number of columns in full matrix
 * @param[in] tile_sz Tile size for base case
 * 
 * @note This function is tail-recursive and could be optimized to iteration
 * @note Stack depth is O(log n), typically < 20 levels even for huge matrices
 */
static void transpose_recursive_impl(
    fft_complex *mat,
    size_t dim,
    size_t row_off,
    size_t col_off,
    size_t stride,
    size_t tile_sz)
{
    // BASE CASE: Small enough to transpose directly
    if (dim <= tile_sz) {
        if (row_off == col_off) {
            // Diagonal block: must transpose in-place
            transpose_diagonal_tile(mat, stride, row_off, dim);
        } else {
            // Off-diagonal block: swap with transpose partner
            swap_tile_complex(mat, stride,
                            row_off, row_off + dim,
                            col_off, col_off + dim,
                            col_off, row_off);
        }
        return;
    }
    
    // RECURSIVE CASE: Divide into quadrants
    const size_t half = dim / 2;
    
    // Step 1: Swap off-diagonal rectangles B ↔ C
    // Rectangle B: rows [0:half], cols [half:dim]
    // Rectangle C: rows [half:dim], cols [0:half]
    process_rectangle_tiled(mat, stride,
                          row_off, row_off + half,           // B's rows
                          col_off + half, col_off + dim,     // B's cols
                          tile_sz);
    
    // Step 2: Recursively transpose top-left quadrant (A)
    transpose_recursive_impl(mat, half,
                           row_off, col_off,
                           stride, tile_sz);
    
    // Step 3: Recursively transpose bottom-right quadrant (D)
    transpose_recursive_impl(mat, dim - half,
                           row_off + half, col_off + half,
                           stride, tile_sz);
}

//==============================================================================
// PUBLIC API IMPLEMENTATION
//==============================================================================

/**
 * @brief Transpose square complex matrix in-place (public interface)
 * 
 * @details
 * This is the main entry point for complex matrix transpose. It computes
 * the optimal tile size and calls the recursive implementation.
 * 
 * **Typical Usage in Four-Step FFT:**
 * @code
 * // After column FFTs and twiddle multiply:
 * fft_transpose_square(data, sqrt_n);
 * // Now ready for row FFTs (columns are now rows!)
 * @endcode
 * 
 * @param[in,out] matrix Pointer to square n×n complex matrix (row-major)
 * @param[in] n Matrix dimension
 */
void fft_transpose_square(fft_complex *matrix, size_t n)
{
    assert(matrix != NULL);
    assert(n > 0);
    
    // Compute optimal tile size based on L1 cache and element size
    const size_t tile_sz = compute_tile_size(sizeof(fft_complex));
    
    // Call recursive implementation starting at origin
    transpose_recursive_impl(matrix, n,
                           0, 0,      // Start at [0, 0]
                           n,         // Stride = full matrix width
                           tile_sz);
}

//==============================================================================
// REAL-VALUED VERSION (for completeness)
//==============================================================================

/**
 * @brief Swap two rectangular tiles in a real (double) matrix
 * @private
 */
static void swap_tile_real(
    double *mat,
    size_t stride,
    size_t r1, size_t r1_end,
    size_t c1, size_t c1_end,
    size_t r2, size_t c2)
{
    const size_t tile_rows = r1_end - r1;
    const size_t tile_cols = c1_end - c1;
    
    for (size_t i = 0; i < tile_rows; i++) {
        for (size_t j = 0; j < tile_cols; j++) {
            size_t idx1 = (r1 + i) * stride + (c1 + j);
            size_t idx2 = (r2 + j) * stride + (c2 + i);
            
            double temp = mat[idx1];
            mat[idx1] = mat[idx2];
            mat[idx2] = temp;
        }
    }
}

/**
 * @brief Transpose a diagonal tile in-place (real version)
 * @private
 */
static void transpose_diagonal_tile_real(
    double *mat,
    size_t stride,
    size_t row_off,
    size_t dim)
{
    for (size_t i = 0; i < dim; i++) {
        for (size_t j = i + 1; j < dim; j++) {
            size_t idx1 = (row_off + i) * stride + (row_off + j);
            size_t idx2 = (row_off + j) * stride + (row_off + i);
            
            double temp = mat[idx1];
            mat[idx1] = mat[idx2];
            mat[idx2] = temp;
        }
    }
}

/**
 * @brief Process rectangle in tiles (real version)
 * @private
 */
static void process_rectangle_tiled_real(
    double *mat,
    size_t stride,
    size_t r_start, size_t r_end,
    size_t c_start, size_t c_end,
    size_t tile_sz)
{
    for (size_t tile_row = r_start; tile_row < r_end; tile_row += tile_sz) {
        for (size_t tile_col = c_start; tile_col < c_end; tile_col += tile_sz) {
            size_t tile_row_end = (tile_row + tile_sz < r_end) ? tile_row + tile_sz : r_end;
            size_t tile_col_end = (tile_col + tile_sz < c_end) ? tile_col + tile_sz : c_end;
            
            swap_tile_real(mat, stride,
                         tile_row, tile_row_end,
                         tile_col, tile_col_end,
                         tile_col, tile_row);
        }
    }
}

/**
 * @brief Recursive transpose (real version)
 * @private
 */
static void transpose_recursive_impl_real(
    double *mat,
    size_t dim,
    size_t row_off,
    size_t col_off,
    size_t stride,
    size_t tile_sz)
{
    if (dim <= tile_sz) {
        if (row_off == col_off) {
            transpose_diagonal_tile_real(mat, stride, row_off, dim);
        } else {
            swap_tile_real(mat, stride,
                         row_off, row_off + dim,
                         col_off, col_off + dim,
                         col_off, row_off);
        }
        return;
    }
    
    const size_t half = dim / 2;
    
    process_rectangle_tiled_real(mat, stride,
                                row_off, row_off + half,
                                col_off + half, col_off + dim,
                                tile_sz);
    
    transpose_recursive_impl_real(mat, half,
                                row_off, col_off,
                                stride, tile_sz);
    
    transpose_recursive_impl_real(mat, dim - half,
                                row_off + half, col_off + half,
                                stride, tile_sz);
}

/**
 * @brief Transpose square real matrix in-place (public interface)
 * 
 * @details
 * Real-valued version for non-complex matrices. Same algorithm,
 * just operates on doubles instead of complex numbers.
 * 
 * @param[in,out] matrix Pointer to square n×n matrix of doubles (row-major)
 * @param[in] n Matrix dimension
 */
void fft_transpose_square_real(double *matrix, size_t n)
{
    assert(matrix != NULL);
    assert(n > 0);
    
    const size_t tile_sz = compute_tile_size(sizeof(double));
    
    transpose_recursive_impl_real(matrix, n,
                                 0, 0,
                                 n,
                                 tile_sz);
}
