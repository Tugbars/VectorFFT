/* kfr_arm.cpp -- the KFR comparator arm (see kfr_arm.h).
 *
 * Wired 2026-09-25, UNTESTED: written against KFR 6's public DFT API
 * (kfr::dft_plan<double>: temp_size, execute(out, in, temp, inverse)) from
 * its documentation, with no KFR installed on the host. The first build
 * with `build.py --kfr` under Clang is the test.
 *
 * KFR's complex type is layout-compatible with two interleaved doubles, so
 * the bench's buffers are passed through a reinterpret_cast, never copied. */
#include "kfr_arm.h"

#include <kfr/dft.hpp>

#include <new>

namespace {
struct kfr_c2c_plan {
    kfr::dft_plan<double> plan;
    explicit kfr_c2c_plan(int N) : plan(static_cast<size_t>(N)) {}
};
}

extern "C" {

void *kfr_c2c_create(int N)
{
    if (N < 1) return nullptr;
    try {
        return new kfr_c2c_plan(N);
    } catch (...) {
        return nullptr;
    }
}

size_t kfr_c2c_temp_size(const void *plan)
{
    return plan ? static_cast<const kfr_c2c_plan *>(plan)->plan.temp_size : 0;
}

void kfr_c2c_forward(const void *plan, const double *in, double *out, unsigned char *temp)
{
    const kfr_c2c_plan *p = static_cast<const kfr_c2c_plan *>(plan);
    p->plan.execute(reinterpret_cast<kfr::complex<double> *>(out),
                    reinterpret_cast<const kfr::complex<double> *>(in),
                    temp, kfr::cfalse);
}

void kfr_c2c_forward_inplace(const void *plan, double *inout, unsigned char *temp)
{
    const kfr_c2c_plan *p = static_cast<const kfr_c2c_plan *>(plan);
    p->plan.execute(reinterpret_cast<kfr::complex<double> *>(inout),
                    reinterpret_cast<const kfr::complex<double> *>(inout),
                    temp, kfr::cfalse);
}

void kfr_c2c_destroy(void *plan)
{
    delete static_cast<kfr_c2c_plan *>(plan);
}

const char *kfr_arm_version(void)
{
    return kfr::library_version();
}

}
