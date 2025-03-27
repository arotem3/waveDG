#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>

#include "wavedg.hpp"
#include "LinSolve/linsolve.hpp"

using namespace dg;

inline auto square(auto x)
{
    return x * x;
}

static inline double force(const double x[2], double omega)
{
    const double r = square(x[0] + 0.7) + square(x[1] + 0.1);
    const double s = square(omega);
    return (s / M_PI) * std::exp(-0.5 * s * r);
}

static int num_elements_per_unit_length(double omega, int order, double K)
{
    double h = std::pow(K / std::pow(omega, order+1), 1.0/order);
    return std::ceil(1.0 / h);
}

static ivec boundary_conditions(const Mesh2D& mesh)
{
    const int nB = mesh.n_edges(FaceType::BOUNDARY);

    // get edge centers and determine if edge is absorbing(bc=0) or reflecting(bc=1)
    auto q = QuadratureRule::quadrature_rule(1); // quadrature rule with collocation point only at center of element

    const double * x_ = mesh.edge_metrics(q, FaceType::BOUNDARY).physical_coordinates();
    auto x = reshape(x_, 2, nB);

    ivec bc(nB);

    for (int e=0; e < nB; ++e)
    {
        const bool left_wall    = std::abs(x(0, e) + 1.0) < 1e-12;
        // const bool right_wall   = std::abs(x(0, e) - 1.0) < 1e-12;
        const bool bottom_wall  = std::abs(x(1, e) + 1.0) < 1e-12;
        // const bool top_wall     = std::abs(x(1, e) - 1.0) < 1e-12;

        if (left_wall || bottom_wall)
            bc(e) = 1;
        else
            bc(e) = 0;
    }

    return bc;
}

class MPIBlas
{
public:
    static double dot(int n, const double * x, int incx, const double * y, int incy)
    {
        double s = 0.0;
        for (int i = 0; i < n; ++i)
            s += x[i * incx] * y[i * incy];
        
    #ifdef WDG_USE_MPI
        double local_s = s;
        MPI_Allreduce(&local_s, &s, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    #endif
    
        return s;
    }

    static double norm(int n, const double * x, int incx)
    {
        double s = 0.0;
        for (int i = 0; i < n; i++)
            s += x[i * incx] * x[i * incx];
    
    #ifdef WDG_USE_MPI
        double local_s = s;
        MPI_Allreduce(&local_s, &s, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    #endif
    
        return std::sqrt(s);
    }

    static void axpy(int n, double alpha, const double * x, int incx, double * y, int incy)
    {
        for (int i = 0; i < n; ++i)
            y[i * incy] += alpha * x[i * incx];
    }

    static void scal(int n, double alpha, double * x, int incx)
    {
        for (int i = 0; i < n; ++i)
            x[i * incx] *= alpha;
    }

    static void copy(int n, const double * x, int incx, double * y, int incy)
    {
        for (int i = 0; i < n; ++i)
            y[i * incy] = x[i * incx];
    }
};

template <typename Operator>
static linsol::SolverResult mpi_gmres(int n, double * x, const Operator &A, const double * b, linsol::gmres_options<double> options = {})
{
    linsol::gmres_solver<double, MPIBlas, std::allocator<double>> solver(options);
    return solver.solve(n, x, A, b);
}

int main(int argc, char ** argv)
{
    MPIEnv mpi(argc, argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    constexpr bool approx_quad = true; // the quadrature rule ends up being exact on rectangular elements
    constexpr double K = 10.0;
    constexpr int max_iter = 1'000;
    constexpr double tol = 1e-6;
    
    const double omega_start = 10, omega_end = 30, omega_delta = 0.5;

    std::ofstream conv_out;
    if (rank == 0)
    {
        conv_out.open("solution/convergence_rate2d.txt");
        conv_out << "omega,order,rho,FP#,GMRES#\n";
        
        std::cout << std::setprecision(3);
        std::cout << std::setw(20) << "omega" << " | "
                  << std::setw(20) << "order" << " | "
                  << std::setw(20) << "#dof" << " | "
                  << std::setw(20) << "rho" << " | "
                  << std::setw(20) << "FP#" << " | "
                  << std::setw(20) << "FP time(sec)" << " | "
                  << std::setw(20) << "GMRES#" << " | "
                  << std::setw(20) << "GMRES time(sec)"
                  << std::endl;
    }

    for (double w = omega_start; w <= omega_end; w += omega_delta)
    {
        const double omega = M_PI * w;
        for (int P : {1, 2})
        {
            const double tic = MPI_Wtime();

            const int nx = 2 * num_elements_per_unit_length(omega, P+1, K);

            Mesh2D mesh;
            if (rank == 0)
                mesh = Mesh2D::uniform_rect(nx, -1.0, 1.0, nx, -1.0, 1.0);
            mesh.distribute("rcb");

            auto basis = QuadratureRule::quadrature_rule(P+1, QuadratureRule::GaussLobatto);
            
            ivec bc = boundary_conditions(mesh);
            WaveHoltz WH(omega, mesh, basis, bc, approx_quad);

            const int n_colloc = basis->n;
            const int global_n_elem = mesh.global_n_elem(); // all elements in mesh
            const int n_elem = mesh.n_elem(); // elements on processor
            const int n_points = n_colloc * n_colloc * n_elem; // local total number of collocation points
            const int n_dof = 3 * n_points; // local number of degrees of freedom

            LinearFunctional2D L(mesh, basis);
            dvec F(n_dof);
            L.action(3, [omega](const double x[2], double * f) -> void {f[0] = force(x, omega); f[1]=0.0; f[2]=0.0;}, F);

            dvec pi0(n_dof);
            WH.pi0(pi0, F);
            const double pi_zero = norm(n_dof, pi0);

            dvec u(n_dof);
            dvec u_prev(n_dof);

            const bool save_iters = (w == 10 || w == 20 || w == 30);
            std::ofstream iter_out;
            if (save_iters && (rank == 0))
            {
                iter_out.open(std::format("solution/iter2d_{}_p{}.txt", (int)w, P));
                iter_out << pi_zero << "\n";
            }

            // fixed-point iteration
            int it = 1;
            double err = pi_zero;
            for (; it <= max_iter; ++it)
            {
                u_prev = u;

                WH.S(u);
                for (int i=0; i < n_dof; ++i)
                    u(i) += pi0(i);
                
                err = error(n_dof, u, u_prev) / pi_zero;

                // if (rank == 0)
                //     std::cout << std::setw(10) << it << " : " << std::scientific << std::setprecision(2) << std::setw(20) << err / pi_zero << "\r" << std::flush;

                if (save_iters && (rank == 0))
                {
                    iter_out << err << std::endl;
                }

                if (err < tol * pi_zero)
                    break;
            }

            if (save_iters && (rank == 0))
            {
                iter_out.close();
            }

            double rho = std::pow(err / pi_zero, 1.0/it);

            const double toc = MPI_Wtime();

            // gmres
            auto IminusS = [&](const double *x, double *y) -> void
            {
                std::copy_n(x, n_dof, y);
                
                WH.S(y);

                for (int i = 0; i < n_dof; ++i)
                    y[i] = x[i] - y[i];
            };

            linsol::gmres_options<double> options;
            options.absolute_tolerance = 1e-12;
            options.relative_tolerance = tol;
            options.verbose = 0;
            options.restart = 1000;
            options.maximum_iterations = 1000;

            zeros(u);
            auto out = mpi_gmres(n_dof, u.data(), IminusS, pi0.data(), options);

            if (save_iters && (rank == 0))
            {
                iter_out.open(std::format("solution/gmres2d_{}_p{}.txt", (int)w, P));
                for (double r : out.residual_norm)
                    iter_out << r << "\n";
                iter_out.close();
            }

            if (rank == 0)
            {
                conv_out << omega << ", " << P << ", " << rho << ", " << it << ", " << out.num_matvec << std::endl;

                std::cout << std::fixed << std::setprecision(2)
                          << std::setw(20) << omega << " | "
                          << std::setw(20) << P << " | "
                          << std::setw(20) << (3 * global_n_elem * n_colloc * n_colloc) << " | "
                          << std::setw(20) << rho << " | "
                          << std::setw(20) << it << " | "
                          << std::setw(20) << (toc - tic) << " | "
                          << std::setw(20) << out.num_matvec << " | "
                          << std::setw(20) << out.time.back()
                          << std::endl;
            }
        }
    }

    return 0;
}
