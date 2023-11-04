#include "linalg.hpp"

extern "C" void dgeevx_(const char * BALANC,
                        const char * JOBVL,
                        const char * JOBVR,
                        const char * SENSE,
                        const int * N,
                        const double * A,
                        const int * LDA,
                        double * WR,
                        double * WI,
                        double * VL,
                        const int * LDVL,
                        double * VR,
                        const int * LDVR,
                        const int * ILO,
                        const int * IHI,
                        double * SCALE,
                        double * ABNRM,
                        double * RCONDE,
                        double * RCONDV,
                        double * WORK,
                        const int * LWORK,
                        int * IWORK,
                        int * INFO);

namespace dg
{
    bool real_eig(int n, double * R, double * e, const double * a)
    {
        if (n == 1)
        {
            *R = 1.0;
            *e = *a;
            return true;
        }

        char bal = 'S';
        char jvl = 'N';
        char jvr = 'V';
        char sense = 'V';
        double * wr = e;
        dg::dvec wi(n);
        double * vl = nullptr;
        int ilo, ihi;
        dg::dvec scale(n);
        double abnrm;
        double rconde;
        double rcondv;
        std::vector<double> dwork(1);
        int lwork = -1;
        std::vector<int> iwork(2*n);
        int info;

        dgeevx_(&bal, &jvl, &jvr, &sense, &n, a, &n, wr, wi.data(), vl, &n, R, &n, &ilo, &ihi, scale.data(), &abnrm, &rconde, &rcondv, dwork.data(), &lwork, iwork.data(), &info);

        lwork = dwork[0];
        dwork.resize(lwork);

        dgeevx_(&bal, &jvl, &jvr, &sense, &n, a, &n, wr, wi.data(), vl, &n, R, &n, &ilo, &ihi, scale.data(), &abnrm, &rconde, &rcondv, dwork.data(), &lwork, iwork.data(), &info);

        if (info != 0)
            return false;

        double max_imag = 0.0;
        for (int i=0; i < n; ++i)
            max_imag = std::max(std::abs(wi(i)), max_imag);
        
        if (max_imag > 0.0)
            return false;
        else
            return true;
    }
} // namespace dg
