#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>

#include "wavedg.hpp"

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
    const int nB = mesh.n_edges(Edge::BOUNDARY);

    // get edge centers and determine if edge is absorbing(bc=0) or reflecting(bc=1)
    auto q = QuadratureRule::quadrature_rule(1); // quadrature rule with collocation point only at center of element

    const double * x_ = mesh.edge_metrics(q, Edge::BOUNDARY).physical_coordinates();
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
        conv_out.open("solution/convergence_rate2d.");
        conv_out << "omega,order,#iter\n";
        
        std::cout << std::setprecision(3);
        std::cout << std::setw(10) << "omega" << " | "
                  << std::setw(10) << "order" << " | "
                  << std::setw(10) << "#dof" << " | "
                  << std::setw(10) << "#iter" << " | "
                  << std::setw(10) << "time(sec)"
                  << std::endl;
    }

    for (double w = omega_start; w <= omega_end; w += omega_delta)
    {
        const double omega = M_PI * w;
        for (int P : {1, 2})
        {
            const double tic = MPI_Wtime();

            const int nx = 2 * num_elements_per_unit_length(omega, P+1, K);

            Mesh2D mesh = Mesh2D::uniform_rect(nx, -1.0, 1.0, nx, -1.0, 1.0);
            mesh.distribute("rcb");

            auto basis = QuadratureRule::quadrature_rule(P+1, QuadratureRule::GaussLobatto);
            
            ivec bc = boundary_conditions(mesh);
            WaveHoltz WH(omega, mesh, basis, bc, approx_quad);

            const int n_colloc = basis->n;
            const int global_n_elem = mesh.global_n_elem(); // all elements in mesh
            const int n_elem = mesh.n_elem(); // elements on processor
            const int n_points = n_colloc * n_colloc * n_elem; // local total number of collocation points
            const int n_dof = 3 * n_points; // local number of degrees of freedom

            LinearFunctional L(mesh, basis);
            dvec F(n_dof);
            L([omega](const double x[2], double * f) -> void {f[0] = force(x, omega); f[1]=0.0; f[2]=0.0;}, F, 3);

            dvec pi0(n_dof);
            WH.pi0(pi0, F);
            const double pi_zero = norm(n_dof, pi0);

            dvec u(n_dof);
            dvec u_prev(n_dof);

            const bool save_iters = (int(2*w) % 20 == 0); // 10pi, 20pi, 30pi
            std::ofstream iter_out;
            if (save_iters && (rank == 0))
            {
                iter_out.open("solution/iter2d_" +std::to_string((int)w/2) + "_" + std::to_string(P) + ".txt");
            }

            bool converged = false;
            int n_iter = max_iter;

            for (int it=1; it <= max_iter; ++it)
            {
                u_prev = u;

                WH.S(u);
                for (int i=0; i < n_dof; ++i)
                    u(i) += pi0(i);
                
                const double err = error(n_dof, u, u_prev) / pi_zero;

                if (not converged)
                {
                    converged = (err < tol);
                    if (converged)
                        n_iter = it;
                }

                if (save_iters && (rank == 0))
                {
                    iter_out << err << std::endl;
                }

                if (err < 1e-14) // stop early once reaching machine precision
                    break;
            }

            const double toc = MPI_Wtime();

            if (rank == 0)
            {
                conv_out << omega << ", " << P << ", " << n_iter << std::endl;

                std::cout << std::setw(10) << omega << " | "
                          << std::setw(10) << P << " | "
                          << std::setw(10) << (global_n_elem * n_colloc * n_colloc * 3) << " | "
                          << std::setw(10) << n_iter << " | "
                          << std::setw(10) << (toc - tic)
                          << std::endl;
            }
        }
    }

    return 0;
}