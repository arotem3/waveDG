#ifndef WDG_PCG_HPP
#define WDG_PCG_HPP

#include <vector>
#include <iostream>
#include <iomanip>

#include "wdg_config.hpp"
#include "Tensor.hpp"
#include "Operator.hpp"
#include "linalg.hpp"

namespace dg
{
    struct solver_results
    {
        bool success;
        int num_iter;
        int num_matvec;
        std::vector<double> res; // res.at(it) is the residual at iteration it

        inline operator bool() {return success;}
    };

    /// @brief solves A * x == b with the preconditioned conjugate gradient method
    /// @param n size of x
    /// @param x length n. On entry, guess for solution (or zeros). On exit, approximate solution to A * x == b
    /// @param A An operator such that A.action(x, y) : y <- A * x. A must be symmetric positive definite.
    /// @param r length n. Initial residual b - A * x. On exit, r is the final residual. If out.success==true, then norm(r) < tol * norm(r0)
    /// @param M Preconditioner. An operator such that M.action(x, y) <- M * x ~ inv(A) * x. M must be symmetric positive definite.
    /// @param max_iter Maxmimum number of iterations before stopping.
    /// @param tol tolerance such that if the relative residual is less than tol, then pcg returns.
    /// @param verbose The amount of information to print to cout. If verbose == 0: silent, 1: progress bar, 2: one line per iteration.
    /// @return summary of iteration
    solver_results pcg(int n, double * x, const Operator * A, double * r, const Operator * M, int max_iter, double tol=1e-6, int verbose=0);
} // namespace dg

#endif
