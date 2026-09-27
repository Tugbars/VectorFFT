/* numtheory.h — the small integer helpers both prime engines need, once.
 *
 *   vfft_is_prime(n)          trial division; n < 2 is not prime
 *   vfft_is_radix_smooth(n)   n factors entirely over the library's PRIME
 *                             radices {2,3,5,7,11,13,17,19} (composite radices
 *                             25, 20, 12, ... build from these); n < 1 is not
 *   vfft_powmod(b, e, m)      b^e mod m, square-and-multiply in long long
 *                             (m < 2^31, so every product fits)
 *   vfft_primitive_root(N)    the SMALLEST primitive root of a prime N: the
 *                             least g with g^((N-1)/p) != 1 (mod N) for every
 *                             prime p | N-1, over the FULL factorization of
 *                             N-1; 0 if none (N not prime)
 *
 * The utilities merge (2026-09-27) folded four copies into this header:
 * prime_dispatch.h (_vfft_is_prime, _vfft_is_radix_smooth),
 * bluestein_calibrator.h (_bcal_is_prime, _bcal_is_radix_smooth), rader.h
 * (_rader_powmod, _rader_find_generator) and il_prime.h (_ilprime_is_prime,
 * _ilprime_powmod, _ilprime_find_generator). The split Rader search factored
 * N-1 over {2..19} only, which is exact where it is used (split Rader is chosen
 * only when N-1 is radix-smooth); on every such N it returns the same root as
 * the full factorization kept here -- proven exhaustively before the merge.
 * vfft_is_radix_smooth(0) used to loop forever (0 % p == 0); it returns 0.
 */
#ifndef VFFT_COMMON_NUMTHEORY_H
#define VFFT_COMMON_NUMTHEORY_H

static inline int vfft_is_prime(int n)
{
    if (n < 2) return 0;
    if ((n & 1) == 0) return n == 2;
    for (int p = 3; (long long)p * p <= n; p += 2)
        if (n % p == 0) return 0;
    return 1;
}

static inline int vfft_is_radix_smooth(int n)
{
    static const int primes[] = {2, 3, 5, 7, 11, 13, 17, 19, 0};
    if (n < 1) return 0;
    for (const int *p = primes; *p; p++)
        while (n % *p == 0) n /= *p;
    return n == 1;
}

static inline long long vfft_powmod(long long b, long long e, long long m)
{
    long long r = 1;
    b %= m;
    while (e > 0) {
        if (e & 1) r = r * b % m;
        b = b * b % m;
        e >>= 1;
    }
    return r;
}

static inline int vfft_primitive_root(int N)
{
    int f[16], nf = 0, r = N - 1;
    for (int p = 2; (long long)p * p <= r; p += (p == 2 ? 1 : 2))
        if (r % p == 0) {
            f[nf++] = p;
            while (r % p == 0) r /= p;
        }
    if (r > 1) f[nf++] = r;
    for (int g = 2; g < N; g++) {
        int ok = 1;
        for (int i = 0; i < nf && ok; i++)
            if (vfft_powmod(g, (N - 1) / f[i], N) == 1) ok = 0;
        if (ok) return g;
    }
    return 0; /* unreachable for prime N */
}

#endif /* VFFT_COMMON_NUMTHEORY_H */
