/*
 * mkl_dfti.h -- the DFTI interface of VectorFFT (docs/roadmap/dfti_shim_design.md).
 *
 * A program written for Intel MKL's DFTI interface builds against this header in place of
 * MKL's: put this directory before MKL's include directory and compile vfft_dfti.c beside the
 * VectorFFT library. The program's descriptors then become VectorFFT plans. Double precision,
 * the interleaved layout; a setting or a geometry that is not served is refused with a status
 * and a printed reason.
 *
 * The names are DFTI's and so are the numeric values of the constants. The functions behind
 * the names are vfft_dfti_*, so MKL can stay linked beside this interface (for BLAS or LAPACK)
 * without a symbol in common.
 */
#ifndef VFFT_DFTI_MKL_DFTI_H
#define VFFT_DFTI_MKL_DFTI_H

#ifdef __cplusplus
extern "C"
{
#endif

/* MKL_LONG: long int, 64 bits under MKL_ILP64; MKL's own definition when its headers came first */
#ifndef MKL_LONG
#ifdef MKL_ILP64
#define MKL_LONG long long int
#else
#define MKL_LONG long int
#endif
#endif

/* ── statuses ──────────────────────────────────────────────────────────── */
#define DFTI_NO_ERROR 0
#define DFTI_MEMORY_ERROR 1
#define DFTI_INVALID_CONFIGURATION 2
#define DFTI_INCONSISTENT_CONFIGURATION 3
#define DFTI_MULTITHREADED_ERROR 4
#define DFTI_BAD_DESCRIPTOR 5
#define DFTI_UNIMPLEMENTED 6
#define DFTI_MKL_INTERNAL_ERROR 7
#define DFTI_NUMBER_OF_THREADS_ERROR 8
#define DFTI_1D_LENGTH_EXCEEDS_INT32 9
#define DFTI_1D_MEMORY_EXCEEDS_INT32 9
#define DFTI_NO_WORKSPACE 11

#define DFTI_MAX_MESSAGE_LENGTH 80
#define DFTI_MAX_NAME_LENGTH 10
#define DFTI_VERSION_LENGTH 198

/* ── the parameters a descriptor carries ───────────────────────────────── */
enum DFTI_CONFIG_PARAM
{
    DFTI_FORWARD_DOMAIN = 0,          /* fixed at create */
    DFTI_DIMENSION = 1,               /* fixed at create */
    DFTI_LENGTHS = 2,                 /* fixed at create */
    DFTI_PRECISION = 3,               /* fixed at create */
    DFTI_FORWARD_SCALE = 4,
    DFTI_BACKWARD_SCALE = 5,
    DFTI_NUMBER_OF_TRANSFORMS = 7,
    DFTI_COMPLEX_STORAGE = 8,
    DFTI_REAL_STORAGE = 9,
    DFTI_CONJUGATE_EVEN_STORAGE = 10,
    DFTI_PLACEMENT = 11,
    DFTI_INPUT_STRIDES = 12,
    DFTI_OUTPUT_STRIDES = 13,
    DFTI_INPUT_DISTANCE = 14,
    DFTI_OUTPUT_DISTANCE = 15,
    DFTI_WORKSPACE = 17,
    DFTI_ORDERING = 18,
    DFTI_TRANSPOSE = 19,              /* read only here */
    DFTI_DESCRIPTOR_NAME = 20,
    DFTI_PACKED_FORMAT = 21,
    DFTI_COMMIT_STATUS = 22,          /* read only */
    DFTI_VERSION = 23,                /* read only; answers a NULL handle too */
    DFTI_NUMBER_OF_USER_THREADS = 26,
    DFTI_THREAD_LIMIT = 27,
    DFTI_DESTROY_INPUT = 28,
    DFTI_FWD_DISTANCE = 58,           /* refused, as in DFTI's C interface */
    DFTI_BWD_DISTANCE = 59            /* refused, as in DFTI's C interface */
};

/* ── the values they take ──────────────────────────────────────────────── */
enum DFTI_CONFIG_VALUE
{
    DFTI_COMMITTED = 30,
    DFTI_UNCOMMITTED = 31,
    DFTI_COMPLEX = 32,
    DFTI_REAL = 33,
    DFTI_SINGLE = 35,                 /* refused: VectorFFT computes in double precision */
    DFTI_DOUBLE = 36,
    DFTI_COMPLEX_COMPLEX = 39,
    DFTI_COMPLEX_REAL = 40,           /* refused (the packed real formats) */
    DFTI_REAL_COMPLEX = 41,           /* refused */
    DFTI_REAL_REAL = 42,              /* split storage: refused; vfft.h serves split data */
    DFTI_INPLACE = 43,
    DFTI_NOT_INPLACE = 44,
    DFTI_ORDERED = 48,
    DFTI_BACKWARD_SCRAMBLED = 49,     /* accepted; the output is in natural order */
    DFTI_ALLOW = 51,
    DFTI_AVOID = 52,
    DFTI_NONE = 53,
    DFTI_CCS_FORMAT = 54,             /* refused */
    DFTI_PACK_FORMAT = 55,            /* refused */
    DFTI_PERM_FORMAT = 56,            /* refused */
    DFTI_CCE_FORMAT = 57
};

/* ── the descriptor ────────────────────────────────────────────────────── */
struct DFTI_DESCRIPTOR;
typedef struct DFTI_DESCRIPTOR *DFTI_DESCRIPTOR_HANDLE;
typedef struct DFTI_DESCRIPTOR DFTI_DESCRIPTOR;        /* older spellings */
#define DFTI_DFT_Desc_struct DFTI_DESCRIPTOR
#define DFTI_Descriptor_struct DFTI_DESCRIPTOR
#define DFTI_Descriptor DFTI_DESCRIPTOR

/* ── the functions ─────────────────────────────────────────────────────── */
/* create: the lengths follow the rank, one MKL_LONG at rank 1, an MKL_LONG array above */
MKL_LONG vfft_dfti_create(DFTI_DESCRIPTOR_HANDLE *desc, enum DFTI_CONFIG_VALUE precision,
                          enum DFTI_CONFIG_VALUE domain, MKL_LONG dimension, ...);
MKL_LONG vfft_dfti_create_d_1d(DFTI_DESCRIPTOR_HANDLE *desc, enum DFTI_CONFIG_VALUE domain, ...);
MKL_LONG vfft_dfti_create_d_md(DFTI_DESCRIPTOR_HANDLE *desc, enum DFTI_CONFIG_VALUE domain, MKL_LONG dimension, ...);
MKL_LONG vfft_dfti_create_s_1d(DFTI_DESCRIPTOR_HANDLE *desc, enum DFTI_CONFIG_VALUE domain, ...);
MKL_LONG vfft_dfti_create_s_md(DFTI_DESCRIPTOR_HANDLE *desc, enum DFTI_CONFIG_VALUE domain, MKL_LONG dimension, ...);
MKL_LONG vfft_dfti_copy(DFTI_DESCRIPTOR_HANDLE from, DFTI_DESCRIPTOR_HANDLE *to);
MKL_LONG vfft_dfti_commit(DFTI_DESCRIPTOR_HANDLE desc);
/* compute: the data pointers follow the placement, one in place, input then output out of place */
MKL_LONG vfft_dfti_compute_forward(DFTI_DESCRIPTOR_HANDLE desc, void *x, ...);
MKL_LONG vfft_dfti_compute_backward(DFTI_DESCRIPTOR_HANDLE desc, void *x, ...);
/* set and get: the value's type follows the parameter (an enumeration is an int, an integer an
 * MKL_LONG, a scale a double, lengths and strides an MKL_LONG array, a name a char array) */
MKL_LONG vfft_dfti_set_value(DFTI_DESCRIPTOR_HANDLE desc, enum DFTI_CONFIG_PARAM param, ...);
MKL_LONG vfft_dfti_get_value(DFTI_DESCRIPTOR_HANDLE desc, enum DFTI_CONFIG_PARAM param, ...);
MKL_LONG vfft_dfti_free(DFTI_DESCRIPTOR_HANDLE *desc);
char *vfft_dfti_error_message(MKL_LONG status);
MKL_LONG vfft_dfti_error_class(MKL_LONG status, MKL_LONG error_class);

/* ── DFTI's names ──────────────────────────────────────────────────────── */
#define DftiCopyDescriptor vfft_dfti_copy
#define DftiCommitDescriptor vfft_dfti_commit
#define DftiComputeForward vfft_dfti_compute_forward
#define DftiComputeBackward vfft_dfti_compute_backward
#define DftiSetValue vfft_dfti_set_value
#define DftiGetValue vfft_dfti_get_value
#define DftiFreeDescriptor vfft_dfti_free
#define DftiErrorMessage vfft_dfti_error_message
#define DftiErrorClass vfft_dfti_error_class
#define DftiCreateDescriptor_d_1d vfft_dfti_create_d_1d
#define DftiCreateDescriptor_d_md vfft_dfti_create_d_md
#define DftiCreateDescriptor_s_1d vfft_dfti_create_s_1d
#define DftiCreateDescriptor_s_md vfft_dfti_create_s_md
/* the create picks the entry by precision and rank, so the lengths reach it with their own type */
#define DftiCreateDescriptor(desc, prec, domain, dim, sizes)                                       \
    (((prec) == DFTI_DOUBLE && (dim) == 1) ? vfft_dfti_create_d_1d((desc), (domain), (sizes))      \
     : ((prec) == DFTI_DOUBLE)            ? vfft_dfti_create_d_md((desc), (domain), (dim), (sizes)) \
                                          : vfft_dfti_create((desc), (prec), (domain), (dim), (sizes)))

#ifdef __cplusplus
}
#endif
#endif /* VFFT_DFTI_MKL_DFTI_H */
