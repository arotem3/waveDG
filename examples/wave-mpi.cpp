#include <iostream>
#include <iomanip>
#include <format>

#include "wavedg.hpp"

using namespace dg;

// formatted file name "%s%05d.%05d" (sol, i, rank), e.g. u00021.00001 when sol="u", i=21, rank=1.
inline static std::string solution_filename(const std::string& sol, int i, int rank)
{
    return std::format("solution/{}{:0>5d}.{:0>5d}", sol, i, rank);
}

// saves solution vector to binary file.
inline static void to_file(const std::string& fname, int n_dof, const double * u)
{
    std::ofstream out(fname, std::ios::out | std::ios::binary);
    out.write(reinterpret_cast<const char*>(u), n_dof * sizeof(double));
    out.close();
}

// specify initial condition.
inline static void initial_conditions(const double x[2], double F[])
{
    F[0] = 0.0;//std::exp(-10.0 * (x[0] * x[0] + x[1] * x[1]));
    F[1] = 0.0;
    F[2] = 0.0;
}

inline static void force(const double t, const double x[2], double F[])
{
    F[0] = 10.0 * std::exp(-100.0 * (x[0] * x[0] + x[1] * x[1])) * std::sin(30 * t);
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
    MPI mpi(argc, argv); // calls MPI_Init. On destruction calls MPI_Finalize.

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    constexpr bool approx_quad = true; // solve with fast quadrature rule?
    constexpr int n_var = 3;

    // specify basis functions. When approx_quad = true, GuassLegendre is more
    // accurate, but GaussLobatto is faster.
    const int n_colloc = 6;
    QuadratureRule::QuadratureType basis_type = QuadratureRule::GaussLobatto;
    auto basis = QuadratureRule::quadrature_rule(n_colloc, basis_type);

    // construct mesh rectangular nx x ny mesh on [x_min, x_max] x [y_min, y_max].
    const int nx = 20, ny = 20;
    const double x_min = -1.0, x_max = 1.0, y_min = -1.0, y_max = 1.0;
    Mesh2D mesh = Mesh2D::uniform_rect(nx, x_min, x_max, ny, y_min, y_max);
    mesh.distribute("rcb");
    
    // mesh statistics
    const int global_n_elem = mesh.global_n_elem(); // all elements in mesh
    const int n_elem = mesh.n_elem(); // elements on processor
    const int n_points = n_colloc * n_colloc * n_elem; // local total number of collocation points
    const int n_dof = n_var * n_points; // local number of degrees of freedom
    const double h = mesh.min_edge_measure(); // shortest length scale

    // time interval: [0, T]
    double t = 0.0; // time variable
    const double T = 2.0;

    constexpr double CFL = 1.0; // Courant-Friedrich-Levy constant
    const double maxvel = 1.0;

    // this dt is optimal for forward Euler, for higher order we can take larger dt
    double dt = CFL / maxvel * h / pow(n_colloc, 2);
    const int nt = std::ceil(T / dt);
    dt = T / nt;

    if (rank == 0)
    {
        const int global_n_dof = n_var * n_colloc * n_colloc * global_n_elem;
        std::cout << "#elements: " << global_n_elem << "\n"
                  << "#DOFs/element: " << n_var << " x " << n_colloc << "^2\n"
                  << "#DOFs: " << global_n_dof << "\n"
                  << "dx: " << h << "\n"
                  << "dt: " << dt << "\n"
                  << "#times steps: " << nt << "\n";
        if (approx_quad)
            std::cout << "quadrature rule: fast (approximate)\n";
        else
            std::cout << "quadrature rule: exact\n";
    }

    // PDE discretization:
    WaveEquation<approx_quad> a(mesh, basis);
    MassMatrix<approx_quad> m(n_var, mesh, basis);
    
    // Boundary conditions
    const ivec _bc = boundary_conditions(mesh);
    WaveBC<approx_quad> bc(mesh, _bc, basis);

    LinearFunctional L(mesh, basis);
    dvec f(n_dof);

    // m * du/dt = a*u + bc*u -> du/dt = m \ (a * u + bc * u).
    auto time_derivative = [&](double * dudt, const double t, const double * u) -> void
    {
        for (int i=0; i < n_dof; ++i)
            dudt[i] = 0.0;
            
        a.action(u, dudt);
        bc.action(u, dudt);

        L([t](const double * x_, double * f_) -> void {force(t,x_,f_);}, f, n_var);
        for (int i=0; i < n_dof; ++i)
            dudt[i] += f(i);

        m.inv(dudt);
    };

    // time integrator
    ode::RungeKutta4 rk(n_dof);

    // set up solution vector.
    Tensor<4,double> u(n_var, n_colloc, n_colloc, n_elem);

    // initial conditions
    Projector project(mesh, m, basis);
    project(initial_conditions, u, n_var);

    // save solution collocation points to file
    auto x = mesh.element_metrics(basis).physical_coordinates();
    to_file(std::format("solution/x.{:0>5d}", rank), 2*n_points, x);
    
    // save initial condition to file
    to_file(solution_filename("u", 0, rank), n_dof, u);
    
    // Time loop
    std::string progress(30, ' ');
    constexpr int skip = 10; // save solution every skip time steps
    for (int it = 1; it <= nt; ++it)
    {
        rk.step(dt, time_derivative, t, u);

        if (it % skip == 0)
            to_file(solution_filename("u", it/skip, rank), n_dof, u);

        progress.at(30*(it-1)/nt) = '#';
        if (rank == 0)
            std::cout << "[" << progress << "]" << std::setw(5) << it << " / " << nt << "\r" << std::flush;
    }
    if (rank == 0)
        std::cout << "\n";
    return 0;
}