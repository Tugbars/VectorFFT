#include "cwtmath.h"
#include <stdlib.h>
#include <math.h>
#include <stdio.h>

/**
 * @brief Rounds a number to the nearest integer toward zero.
 * @param x The input number to round.
 * @return The rounded value (floor for positive, ceil for negative).
 */
static double fix(double x) {
    return (x >= 0.0) ? floor(x) : ceil(x);
}

/**
 * @brief Rounds a number to the nearest integer.
 * @param x The input number to round.
 * @return The nearest integer, rounded up if x >= x.5.
 */
static int nint(double x) {
    return (int)(x + 0.49999);
}

/**
 * @brief Computes the forward Non-Standard Fast Fourier Transform (NSFFT).
 * @param obj The FFT object containing configuration (N, sgn, etc.).
 * @param inp Input data array of complex numbers.
 * @param oup Output data array for transformed results.
 * @param lb Lower bound of the interval.
 * @param ub Upper bound of the interval.
 * @param w Output array for frequency weights.
 * @note The length N must be a power of 2, or the function will exit with an error.
 */
static void nsfft_fd(fft_object obj, fft_data *inp, fft_data *oup, double lb, double ub, double *w) {
    int N = obj->N; // Number of samples
    int L = N / 2;  // Half the number of samples
    int i, j;

    // Validate that N is a power of 2
    if (divideby(N, 2) == 0) {
        fprintf(stderr, "Error: NSFFT length must be a power of 2\n");
        exit(EXIT_FAILURE);
    }

    // Allocate temporary arrays for real and imaginary parts
    double *temp1 = (double *)malloc(sizeof(double) * L);
    double *temp2 = (double *)malloc(sizeof(double) * L);
    if (!temp1 || !temp2) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        free(temp1);
        free(temp2);
        exit(EXIT_FAILURE);
    }

    // Compute frequency weights: w[i] = j / (2 * (ub - lb)), where j ranges from -N to N-2
    double delta = (ub - lb) / N; // Interval step size
    double den = 2.0 * (ub - lb); // Denominator for weights
    for (i = 0, j = -N; i < N; ++i, j += 2) {
        w[i] = (double)j / den;
    }

    // Execute standard FFT
    fft_exec(obj, inp, oup);

    // Store first L elements of output
    for (i = 0; i < L; ++i) {
        temp1[i] = oup[i].re;
        temp2[i] = oup[i].im;
    }

    // Shift output: move elements [L, N-1] to [0, N-L-1]
    for (i = 0; i < N - L; ++i) {
        oup[i].re = oup[i + L].re;
        oup[i].im = oup[i + L].im;
    }

    // Place stored elements at the end: [N-L, N-1]
    for (i = 0; i < L; ++i) {
        oup[N - L + i].re = temp1[i];
        oup[N - L + i].im = temp2[i];
    }

    // Apply phase shift and scaling for non-standard interval
    double plb = PI2 * lb; // 2 * pi * lower bound
    for (i = 0; i < N; ++i) {
        double tempr = oup[i].re;
        double tempi = oup[i].im;
        double theta = w[i] * plb; // Phase angle
        oup[i].re = delta * (tempr * cos(theta) + tempi * sin(theta));
        oup[i].im = delta * (tempi * cos(theta) - tempr * sin(theta));
    }

    // Clean up
    free(temp1);
    free(temp2);
}

/**
 * @brief Computes the backward (inverse) Non-Standard Fast Fourier Transform (NSFFT).
 * @param obj The FFT object containing configuration (N, sgn, etc.).
 * @param inp Input data array of complex numbers (frequency domain).
 * @param oup Output data array for transformed results (time domain).
 * @param lb Lower bound of the interval.
 * @param ub Upper bound of the interval.
 * @param t Output array for time points.
 * @note The length N must be a power of 2, or the function will exit with an error.
 */
static void nsfft_bk(fft_object obj, fft_data *inp, fft_data *oup, double lb, double ub, double *t) {
    int N = obj->N; // Number of samples
    int L = N / 2;  // Half the number of samples
    int i, j;

    // Validate that N is a power of 2
    if (divideby(N, 2) == 0) {
        fprintf(stderr, "Error: NSFFT length must be a power of 2\n");
        exit(EXIT_FAILURE);
    }

    // Allocate temporary arrays and input buffer
    double *temp1 = (double *)malloc(sizeof(double) * L);
    double *temp2 = (double *)malloc(sizeof(double) * L);
    double *w = (double *)malloc(sizeof(double) * N);
    fft_data *inpt = (fft_data *)malloc(sizeof(fft_data) * N);
    if (!temp1 || !temp2 || !w || !inpt) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        free(temp1);
        free(temp2);
        free(w);
        free(inpt);
        exit(EXIT_FAILURE);
    }

    // Compute frequency weights
    double delta = (ub - lb) / N; // Interval step size
    double den = 2.0 * (ub - lb); // Denominator for weights
    for (i = 0, j = -N; i < N; ++i, j += 2) {
        w[i] = (double)j / den;
    }

    // Apply inverse phase shift and scaling
    double plb = PI2 * lb; // 2 * pi * lower bound
    for (i = 0; i < N; ++i) {
        double theta = w[i] * plb; // Phase angle
        inpt[i].re = (inp[i].re * cos(theta) - inp[i].im * sin(theta)) / delta;
        inpt[i].im = (inp[i].im * cos(theta) + inp[i].re * sin(theta)) / delta;
    }

    // Store first L elements of input
    for (i = 0; i < L; ++i) {
        temp1[i] = inpt[i].re;
        temp2[i] = inpt[i].im;
    }

    // Shift input: move elements [L, N-1] to [0, N-L-1]
    for (i = 0; i < N - L; ++i) {
        inpt[i].re = inpt[i + L].re;
        inpt[i].im = inpt[i + L].im;
    }

    // Place stored elements at the end: [N-L, N-1]
    for (i = 0; i < L; ++i) {
        inpt[N - L + i].re = temp1[i];
        inpt[N - L + i].im = temp2[i];
    }

    // Execute standard FFT
    fft_exec(obj, inpt, oup);

    // Compute time points: t[i] = lb + i * delta
    for (i = 0; i < N; ++i) {
        t[i] = lb + i * delta;
    }

    // Clean up
    free(w);
    free(temp1);
    free(temp2);
    free(inpt);
}

/**
 * @brief Executes the Non-Standard FFT (forward or backward based on obj->sgn).
 * @param obj The FFT object containing configuration (N, sgn, etc.).
 * @param inp Input data array of complex numbers.
 * @param oup Output data array for transformed results.
 * @param lb Lower bound of the interval.
 * @param ub Upper bound of the interval.
 * @param w Output array for frequency weights (forward) or time points (backward).
 */
void nsfft_exec(fft_object obj, fft_data *inp, fft_data *oup, double lb, double ub, double *w) {
    if (obj->sgn == 1) {
        nsfft_fd(obj, inp, oup, lb, ub, w); // Forward transform
    } else if (obj->sgn == -1) {
        nsfft_bk(obj, inp, oup, lb, ub, w); // Backward transform
    } else {
        fprintf(stderr, "Error: Invalid FFT sign value\n");
        exit(EXIT_FAILURE);
    }
}

/**
 * @brief Computes the gamma function for real numbers.
 * @param x The input value.
 * @return The gamma function value, or xinf (infinity) for overflow/pole cases.
 * @note Based on W.J. Cody's Fortran code (http://www.netlib.org/specfun/gamma).
 * @see References:
 * - W.J. Cody, "An Overview of Software Development for Special Functions", Lecture Notes in Mathematics, 506, 1976.
 * - Hart et al., "Computer Approximations", Wiley and Sons, 1968.
 */
double cwt_gamma(double x) {
    // Constants
    const double spi = 0.9189385332046727417803297; // log(sqrt(2*pi))
    const double pi = 3.1415926535897932384626434;
    const double xmax = 171.624e+0; // Maximum x before overflow
    const double xinf = 1.79e308;   // Machine infinity
    const double eps = 2.22e-16;    // Machine epsilon
    const double xninf = 1.79e-308; // Minimum positive number

    // Coefficients for rational approximation (1 <= x <= 2)
    static const double num[8] = {
        -1.71618513886549492533811e+0, 2.47656508055759199108314e+1,
        -3.79804256470945635097577e+2, 6.29331155312818442661052e+2,
        8.66966202790413211295064e+2, -3.14512729688483675254357e+4,
        -3.61444134186911729807069e+4, 6.64561438202405440627855e+4
    };
    static const double den[8] = {
        -3.08402300119738975254353e+1, 3.15350626979604161529144e+2,
        -1.01515636749021914166146e+3, -3.10777167157231109440444e+3,
        2.25381184209801510330112e+4, 4.75584627752788110767815e+3,
        -1.34659959864969306392456e+5, -1.15132259675553483497211e+5
    };

    // Coefficients for Hart's minimax approximation (x >= 12)
    static const double c[7] = {
        -1.910444077728e-03, 8.4171387781295e-04, -5.952379913043012e-04,
        7.93650793500350248e-04, -2.777777777777681622553e-03,
        8.333333333333333331554247e-02, 5.7083835261e-03
    };

    double y = x; // Working copy of input
    int swi = 0;  // Switch for negative x handling
    double fact = 1.0; // Factor for reflection formula
    int n = 0;    // Integer part for x > 1
    double oup;   // Output value

    // Handle negative x using reflection formula
    if (y < 0.0) {
        y = -x;
        double yi = fix(y); // Integer part toward zero
        double oup_temp = y - yi; // Fractional part
        if (oup_temp != 0.0) {
            if (yi != fix(yi * 0.5) * 2.0) {
                swi = 1; // Odd integer part
            }
            fact = -pi / sin(pi * oup_temp); // Reflection formula factor
            y += 1.0;
        } else {
            return xinf; // Pole at negative integers
        }
    }

    // Handle small x (near zero)
    if (y < eps) {
        if (y >= xninf) {
            oup = 1.0 / y;
        } else {
            return xinf; // Underflow
        }
    }
    // Handle 0 < x < 12 using rational approximation
    else if (y < 12.0) {
        double yi = y;
        double z;
        if (y < 1.0) {
            z = y;
            y += 1.0;
        } else {
            n = (int)(y - 1.0);
            y -= (double)n;
            z = y - 1.0;
        }
        double nsum = 0.0, dsum = 1.0;
        for (int i = 0; i < 8; ++i) {
            nsum = (nsum + num[i]) * z;
            dsum = dsum * z + den[i];
        }
        oup = nsum / dsum + 1.0;
        if (yi < y) {
            oup /= yi;
        } else if (yi > y) {
            for (int i = 0; i < n; ++i) {
                oup *= y;
                y += 1.0;
            }
        }
    }
    // Handle x >= 12 using minimax approximation
    else {
        if (y <= xmax) {
            double y2 = y * y;
            double sum = c[6];
            for (int i = 0; i < 6; ++i) {
                sum = sum / y2 + c[i];
            }
            sum = sum / y - y + spi;
            sum += (y - 0.5) * log(y);
            oup = exp(sum);
        } else {
            return xinf; // Overflow
        }
    }

    // Apply reflection formula for negative x
    if (swi) {
        oup = -oup;
    }
    if (fact != 1.0) {
        oup = fact / oup;
    }

    return oup;
}
