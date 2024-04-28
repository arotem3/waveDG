#include "pcg.hpp"

class ProgressBar
{
public:
    ProgressBar(int nt_) : it{0}, nt{nt_}, progress(30, ' ') {}

    void operator++()
    {
        it = std::min(it+1, nt-1);
        for (int i=0; nt*i < 30*(it-1); ++i)
            progress.at(i) = '#';
    }

    const std::string& get() const
    {
        return progress;
    }

private:
    int it;
    int nt;
    std::string progress;
};

namespace dg
{
    solver_results pcg(int n, double * x, const Operator * A, double * r, const Operator * M, int max_iter, double tol, int verbose)
    {
        double r0 = norm(n, r);

        solver_results out;
        out.success = false;
        out.num_iter = 0;
        out.num_matvec = 0;
        out.res.push_back(r0);

        dvec p(n), z(n);
        double rho_prev, rho;

        ProgressBar bar(max_iter);

        int it;
        for (it = 0; it < max_iter; ++it)
        {
            if (it == 0)
            {
                if (M) M->action(r, p);
                else copy(n, r, p);

                rho = dot(n, r, p);
            }
            else
            {
                if (M)
                {
                    M->action(r, z);
                    rho = dot(n, r, z);
                    const double beta = rho / rho_prev;
                    axpby(n, 1.0, z, beta, p); // p <- z + beta * p
                }
                else
                {
                    rho = dot(n, r, r);
                    const double beta = rho / rho_prev;
                    axpby(n, 1.0, r, beta, p); // p <- r + beta * p
                }
            }

            if (rho < 0)
                wdg_error("pcg() error: System or preconditioner is not positive definite.");

            A->action(p, z);
            const double alpha = rho / dot(n, p, z);
            axpby(n,  alpha, p, 1.0, x); // x <- x + alpha * p
            axpby(n, -alpha, z, 1.0, r); // r <- r - alpha * A * p

            rho_prev = rho;
            
            const double res = std::sqrt(rho);
            out.res.push_back(res);

            if (verbose == 1)
            {
                ++bar;
                std::cout << "[" << bar.get() << "] || iteration " << std::setw(10) << it << " / " << max_iter << " || rel. res. = " << std::setw(10) << res/r0 << "\r" << std::flush;
            }
            else if (verbose == 2)
                std::cout << "Iteration " << std::setw(10) << it << " / " << max_iter << " || rel. res. = " << std::setw(20) << res/r0 << std::endl;

            if (res < tol * r0)
            {
                out.success = true;
                break;
            }
        }

        if (verbose == 1)
            std::cout << std::endl;
        if (verbose)
        {
            std::cout << "After " << it << " iterations, PCG achieved rel. residual of " << out.res.back()/r0 << std::endl;
            if (out.success)
                std::cout << "PCG successfully converged within desired tolerance." << std::endl;
            else
                std::cout << "PCG failed to converge within desired tolerance." << std::endl;
        }

        out.num_iter = it;
        out.num_matvec = it;
        return out;
    }
} // namespace dg
