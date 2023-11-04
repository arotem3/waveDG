#include <iostream>
#include <fstream>
#include <iomanip>

#include "wavedg.hpp"

using namespace dg;

inline static std::string solution_filename(const std::string& sol, int i)
{
    return "solution/" + sol + std::to_string(i) + ".dat";
}

inline static void to_file(const std::string& fname, int n_dof, const double * u)
{
    std::ofstream out(fname, std::ios::out | std::ios::binary);
    out.write(reinterpret_cast<const char*>(u), n_dof * sizeof(double));
    out.close();
}

inline static void initial_conditions(const double x[2], double F[])
{
    *F = std::exp(-10.0 * (x[0] * x[0] + x[1] * x[1]));
}

int main()
{
    // solve with approximate quadrature rule?
    constexpr bool approx_quad = false;

    // PDE: u_t + c[0] * u_x + c[1] * u_y == 0.
    constexpr int n_var = 1;
    const double c[] = {1.0, 2.0};

    // specify basis functions. When approx_quad = true, GuassLegendre is more
    // accurate, but GaussLobatto is faster.
    const int n_colloc = 5;
    QuadratureRule::QuadratureType qtype = QuadratureRule::GaussLobatto;
    auto basis = QuadratureRule::quadrature_rule(n_colloc, qtype);

    // construct mesh rectangular nx x ny mesh on [x_min, x_max] x [y_min, y_max].
    const int nx = 10, ny = 10;
    const double x_min = -1.0, x_max = 1.0, y_min = -1.0, y_max = 1.0;
    Mesh2D mesh = Mesh2D::uniform_rect(nx, x_min, x_max, ny, y_min, y_max);
    
    const int n_elem = mesh.n_elem();
    const double h = mesh.min_edge_measure(); // shortest length scale

    // time interval: [0, T]
    double t = 0.0; // time variable
    const double T = 1.0;

    constexpr double CFL = 1.0; // Courant-Friedrich-Levy constant
    const double maxvel = std::max(c[0], c[1]);

    double dt = CFL / maxvel * h / pow(n_colloc-1, 2);
    const int nt = std::ceil(T / dt);
    dt = T / nt;
    
    // determine degrees of freedom.
    const int n_points = n_colloc * n_colloc * n_elem;
    const int n_dof = n_var * n_points;

    std::cout << "#elements: " << n_elem << "\n"
              << "#DOFs/element: " << n_var << " x " << n_colloc << "^2\n"
              << "#DOFs: " << n_dof << "\n"
              << "dx: " << h << "\n"
              << "dt: " << dt << "\n"
              << "#times steps: " << nt << "\n";
    if (approx_quad)
        std::cout << "using approximate quadrature rule...\n";

    // initialize pde discretization.
    // Advection: a*u -> -(c u, grad v) - <(c u)*, v>
    // MassMatrix m*u -> (u, v)
    // BC: bc*u -> specify u == 0 outside domain.
    Advection<approx_quad> a(n_var, mesh, basis, c, true);
    MassMatrix<approx_quad> m(n_var, mesh, basis);
    AdvectionHomogeneousBC<approx_quad> bc(n_var, mesh, basis, c, true);

    // m * du/dt = a*u + bc*u -> du/dt = m \ (a * u + bc * u).
    auto time_derivative = [&](double * dudt, const double t, const double * u) -> void
    {
        for (int i=0; i < n_dof; ++i)
            dudt[i] = 0.0;
            
        a.action(u, dudt);
        bc.action(u, dudt);
        m.inv(dudt);
    };

    // time integrator
    ode::RungeKutta2 rk(n_dof);

    // set up solution vector.
    Tensor<4,double> u(n_var, n_colloc, n_colloc, n_elem);

    // initial conditions
    Projector project(mesh, m, basis);
    project(initial_conditions, u, n_var);

    // save solution collocation points to file
    auto x = mesh.element_physical_coordinates(basis);
    to_file("solution/x.dat", 2*n_points, x);
    
    // save initial condition to file
    to_file(solution_filename("u", 0), n_dof, u);
    
    // Time loop
    std::string progress(30, ' ');
    constexpr int skip = 10; // save solution every skip time steps
    for (int it = 1; it <= nt; ++it)
    {
        rk.step(dt, time_derivative, t, u);

        if (it % skip == 0)
            to_file(solution_filename("u", it/skip), n_dof, u);

        progress.at(30*(it-1)/nt) = '#';
        std::cout << "[" << progress << "]" << std::setw(5) << it << " / " << nt << "\r" << std::flush;
    }
    std::cout << "\n";

    return 0;
}