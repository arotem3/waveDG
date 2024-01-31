#include <iostream>
#include <iomanip>
#include <format>

#include "wavedg.hpp"

using namespace dg;

// saves solution vector to binary file.
inline static void to_file(const std::string& fname, int n_dof, const double * u)
{
    std::ofstream out(fname, std::ios::out | std::ios::binary);
    out.write(reinterpret_cast<const char*>(u), n_dof * sizeof(double));
    out.close();
}

inline static void force(const double x[2], double F[])
{
    double w = std::pow(10.0 * M_PI, 2);
    double r = std::pow(x[0] + 0.7, 2) + std::pow(x[1] + 0.1, 2);
    F[0] = w/M_PI * std::exp(-w * r);
    F[1] = 0.0;
    F[2] = 0.0;
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

    constexpr bool approx_quad = true;
    constexpr int n_var = 3;

    const int n_colloc = 5;
    QuadratureRule::QuadratureType basis_type = QuadratureRule::GaussLobatto;
    auto basis = QuadratureRule::quadrature_rule(n_colloc, basis_type);

    const int nx = 60, ny = 60;
    const double x_min = -1.0, x_max = 1.0, y_min = -1.0, y_max = 1.0;
    Mesh2D mesh = Mesh2D::uniform_rect(nx, x_min, x_max, ny, y_min, y_max);
    mesh.distribute("rcb");

    const int global_n_elem = mesh.global_n_elem(); // all elements in mesh
    const int n_elem = mesh.n_elem(); // elements on processor
    const int n_points = n_colloc * n_colloc * n_elem; // local total number of collocation points
    const int n_dof = n_var * n_points; // local number of degrees of freedom
    const double h = mesh.min_edge_measure(); // shortest length scale

    if (rank == 0)
    {
        const int global_n_dof = n_var * n_colloc * n_colloc * global_n_elem;
        std::cout << "#elements: " << global_n_elem << "\n"
                  << "#DOFs/element: " << n_var << " x " << n_colloc << "^2\n"
                  << "#DOFs: " << global_n_dof << "\n"
                  << "dx: " << h << "\n";
        if (approx_quad)
            std::cout << "quadrature rule: fast (approximate)\n";
        else
            std::cout << "quadrature rule: exact\n";
    }

    const double omega = 10.0 * M_PI;
    const ivec bc = boundary_conditions(mesh);
    WaveHoltz WH(omega, mesh, basis, bc, approx_quad);

    LinearFunctional L(mesh, basis);
    dvec f(n_dof);

    L(force, f, 3);

    dvec pi0(n_dof);
    WH.pi0(pi0, f);

    const double pi_zero = norm(n_dof, pi0);

    dvec uhat(n_dof);
    dvec uprev(n_dof);

    const int maxit = 1'000;
    const double tol = 1e-6;

    std::string progress(30, ' ');
    std::cout << std::setprecision(3) << std::scientific;
    for (int it = 1; it <= maxit; ++it)
    {
        uprev = uhat;

        // u <- S * u + pi0
        WH.S(uhat);

        for (int i=0; i < n_dof; ++i)
            uhat(i) += pi0(i);

        const double err = error(n_dof, uhat, uprev) / pi_zero;

        progress.at(30*(it-1)/maxit) = '#';
        if (rank == 0)
        {
            std::cout << "[" << progress << "]"
                      << std::setw(5) << it << " / " << maxit
                      << " | err = " << std::setw(10) << err
                      << "\r" << std::flush;
        }
        
        if (err < tol)
            break;
    }

    if (rank == 0)
        std::cout << std::endl;

    uprev = uhat;
    WH.postprocess(uhat, uprev);

    auto x = mesh.element_metrics(basis).physical_coordinates();
    to_file(std::format("solution/x.{:0>5d}", rank), 2*n_points, x);

    to_file(std::format("solution/u.{:0>5d}", rank), 2*n_points, uhat);
    
    return 0;
}