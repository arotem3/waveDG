#include "eig.hpp"

extern "C" void dsyev_(const char *, const char *, const int *, double *, const int *, double *, double *, int *, int *);

static inline double square(double x)
{
    return x * x;
}

static bool eig2(double * R, double * e, const double * a)
{
    double D = square(a[0]) - 2.0*a[0]*a[3] + 4.0*square(a[1]) + square(a[3]);

    if (D < 0)
        return false;

    D = std::sqrt(D);
    e[0] = 0.5 * (a[0] + a[3] - D);
    e[1] = 0.5 * (a[0] + a[3] + D);

    double x = -(-a[0] + a[3] + D) / (2.0 * a[2]);
    double r = std::hypot(x, 1.0);
    R[0] = x/r;
    R[1] = 1.0/r;

    x = -(-a[0] + a[3] - D) / (2.0 * a[2]);
    r = std::hypot(x, 1.0);
    R[2] = x/r;
    R[3] = 1.0/r;

    return true;
}

static bool _eigdecomp(int n, double * R, double * e, const double * a)
{
    for (int i = 0; i < n*n; ++i)
        R[i] = a[i];

    char jobz = 'V';
    char uplo = 'U';

    std::vector<double> work(1);
    int lwork = -1;

    int info;

    dsyev_(&jobz, &uplo, &n, R, &n, e, work.data(), &lwork, &info);

    lwork = work[0];
    work.resize(lwork);

    dsyev_(&jobz, &uplo, &n, R, &n, e, work.data(), &lwork, &info);

    return (info == 0);
}

namespace dg
{
    bool eig(int n, double * R, double * e, const double * a)
    {
        switch (n)
        {
        case 1:
            *R = 1.0;
            *e = *a;
            return true;
        case 2:
            return eig2(R, e, a);
        default:
            return _eigdecomp(n, R, e, a);
        }
    }
} // namespace dg
