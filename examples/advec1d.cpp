/* 


 */

#include <iostream>
#include <iomanip>
#include <fstream>
#include <format>

#include "wavedg.hpp"

using namespace dg;

inline static void initial_conditions(const double x[], double F[])
{
    // *F = std::exp(-30.0 * x[0] * x[0]);
    *F = std::sin(4 * M_PI * x[0]);
    // *F = std::sin(M_PI * x[0] / 2);
}

inline static void to_file(const std::string& fname, int n_dof, const double * u)
{
    std::ofstream out(fname, std::ios::out | std::ios::binary);
    out.write(reinterpret_cast<const char*>(u), n_dof * sizeof(double));
    out.close();
}

inline static void speed(const double x[], double F[])
{
    *F = 1.0 + 0.5 * std::sin(4 * M_PI * (*x));
    // *F = 1.0;
}

int main(int argc, char ** argv)
{
    constexpr bool approx_quad = false;

    const int n_colloc = 10;
    QuadratureRule::QuadratureType basis_type = QuadratureRule::GaussLegendre;
    auto basis = QuadratureRule::quadrature_rule(n_colloc, basis_type);

    const int n_elem = 10;
    Mesh1D mesh = Mesh1D::uniform_mesh(n_elem, -1.0, 1.0);

    const int n_points = n_colloc * n_elem;
    const int n_dof = n_points;
    const double h = mesh.min_h();

    // Mass Matrix
    MassMatrix<approx_quad> m(1, mesh, basis);

    // Project variable coefficient
    auto quad = QuadratureRule::quadrature_rule(n_colloc/2 + 5);
    LinearFunctional LF(1, mesh, basis, quad);
    dmat c(n_colloc, n_elem);
    LF(speed, c);
    m.inv(c);

    // max speed
    const double max_c = *std::max_element(c.begin(), c.end(), [](double a, double b)->bool{return std::abs(a) < std::abs(b);});

    double t = 0.0;
    const double T = 4.0;
    
    const double CFL = 1.0 / std::pow(n_colloc, 2);

    double dt = CFL / max_c * h;
    const int nt = std::ceil(T / dt);
    dt = T / nt;

    std::cout << "#elements: " << n_elem << "\n"
              << "#DOFs/element: " << n_colloc << "\n"
              << "#DOFs: " << n_dof << "\n"
              << "dx: " << h << "\n"
              << "dt: " << dt << "\n"
              << "#times steps: " << nt << "\n";

    // PDE discretization
    Advection<approx_quad> a(1, mesh, basis, c, false);
    // AdvectionHomogeneousBC<approx_quad> bc(1, mesh, basis, c, false);
    PeriodicBC1d bc(1, mesh, basis, c, false);

    // m * du/dt = a*u + bc*u -> du/dt = m \ (a*u + bc*u).
    auto time_derivative = [&](double * dudt, const double t, const double * u) -> void
    {
        for (int i = 0; i < n_dof; ++i)
            dudt[i] = 0.0;
        
        a.action(u, dudt);
        bc.action(u, dudt);
        m.inv(dudt);
    };

    // time integrator
    ode::RungeKutta4 rk(n_dof);

    // set up solution vector.
    dmat u(n_colloc, n_elem);

    // Project Initial Conditions
    LF(initial_conditions, u);
    m.inv(u);

    // save solution collocation points to file
    auto _x = mesh.element_metrics(basis).physical_coordinates();
    auto x = reshape(_x, n_colloc, n_elem);

    to_file("solution/x.00000", n_points, x);

    // save initial conditions to file
    to_file(std::format("solution/u{:0>5d}.00000", 0), n_dof, u);

    // Time loop
    std::string progress(30, ' ');
    constexpr int skip = 10; // save solution every skip time steps
    for (int it = 1; it <= nt; ++it)
    {
        rk.step(dt, time_derivative, t, u);

        if (it % skip == 0)
            to_file(std::format("solution/u{:0>5d}.00000", it/skip), n_dof, u);

        progress.at(30*(it-1)/nt) = '#';
        std::cout << "[" << progress << "]" << std::setw(5) << it << " / " << nt << "\r" << std::flush;
    }
    std::cout << std::endl;

    return 0;
}