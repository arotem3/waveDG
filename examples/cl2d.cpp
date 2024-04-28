/** @file cl2d.cpp
 *  @brief Example driver for solving Euler's equation for compressible fluid dynamics in 2D
 * 
 * This file is a driver for solving Euler's equation:
 * $$\rho_t + (\rho u)_x + (\rho v)_y = 0,$$
 * $$(\rho u)_t + (\rho u^2 + p)_x + (\rho u v)_y = 0,$$
 * $$(\rho v)_t + (\rho u v)_x + (\rho v^2 + p)_y = 0,$$
 * $$E_t + (u(E + p))_x + (v(E + p))_y = 0.$$
 * 
 * Here \f$p = \frac{1}{\gamma-1} p + \frac{1}{2} \rho (u^2 + v^2)\f$ is the pressure
 * and \f$\gamma=1.4\f$ is the ideal gas constant.
 * 
 * To compile & run this program, first compile the library in serial mode:
 * 
 * `cmake . -D WDG_USE_MPI=OFF`
 * `make wavedg -j`
 * 
 * Then compile this file with
 * 
 * `make cl2d`
 * 
 * And run with
 * 
 * `./examples/cl2d`
 * 
 * The program will write the collocation points and solution
 * values to `solution/x.00000` and `solution/u%05d.00000`, respectively, in binary format.
 * Where the number on u is the time step; e.g. at time step 10 we write `solution/u00010.00000` 
 * and at time step 555 we write `solution/u00555.00000`, etc.
 */

#include <iostream>
#include <iomanip>
#include <format>

#include "wavedg.hpp"
#include "examples.hpp"

using namespace dg;

/// specify initial condition.
inline static void initial_conditions(const double x[2], double q[4])
{
    constexpr double gamma = 1.4;
    const double r = std::pow(x[0] - 5.0, 2) + x[1] * x[1];
    const double b = 2.5 * std::exp(1.0 - r) / M_PI;
    const double u = 1.0 - b * x[1];
    const double v = b * (x[0] - 5.0);
    const double rho = std::pow(1.0 - (gamma-1.0)/(4.0*gamma)*b*b, 1.0/(gamma-1.0));
    const double p = std::pow(rho, gamma);

    q[0] = rho;
    q[1] = rho * u;
    q[2] = rho * v;
    q[3] = p/(gamma-1.0) + 0.5 * rho * (u*u + v*v);
}

// Euler's equation for compressible fluid dynamics
inline static void F(const double x[2], const double q[4], double f_[8])
{
    double * f = f_, * g = f_+4;

    constexpr double gamma = 1.4;
    
    const double rho = q[0];
    const double u   = q[1] / q[0];
    const double v   = q[2] / q[0];
    const double E   = q[3];

    const double p = (gamma - 1.0) * (E - 0.5 * rho * (u * u + v * v));

    f[0] = rho * u;
    f[1] = rho * u * u + p;
    f[2] = rho * u * v;
    f[3] = u * (E + p);

    g[0] = rho * v;
    g[1] = rho * u * v;
    g[2] = rho * v * v + p;
    g[3] = v * (E + p);
}

// local speed of propagation
inline static double speed(const double q[4])
{
    constexpr double gamma = 1.4;

    const double rho = q[0];
    const double u   = q[1] / q[0];
    const double v   = q[2] / q[0];
    const double E   = q[3];

    const double p = (gamma - 1.0) * (E - 0.5 * rho * (u * u + v * v));

    return std::hypot(u, v) + std::sqrt(gamma * p / rho);
}

// Local Lax-Freidrich's flux
inline static void numerical_flux(const double x[2], const double n[2], const double qL[4], const double qR[4], double fh[4])
{
    double fL[8], fR[8];
    F(x, qL, fL);
    F(x, qR, fR);

    const double c = std::max(speed(qL), speed(qR));

    for (int d = 0; d < 4; ++d)
    {
        fh[d] = 0.5 * n[0] * (fL[d] + fR[d]) + 0.5 * n[1] * (fL[4+d] + fR[4+d]) + 0.5 * c * (qL[d] - qR[d]);
    }
}

constexpr static double max_speed()
{
    return 6.0; // this is the maximum for this particular initial conditions in the domain [0,10]x[-6,6] and t<2
}

int main(int argc, char ** argv)
{
    // approx_quad == true ==> compute integrals using quadrature rule corresponding to the Lagrange basis collocation points.
    // approx_quad == false ==> compute integrals on higher order quadrature rule (automatically determined).
    constexpr bool approx_quad = false;

    constexpr int n_var = 4; // dimension of Euler's equations

    // Specify basis functions in terms of 1D quadrature rule. Basis functions
    // are tensor product of 1D Lagrange interpolating polynomials on Gauss
    // quadrature rule. The order of the DG discretization is n_colloc - 1/2.
    const int n_colloc = 5;
    QuadratureRule::QuadratureType basis_type = QuadratureRule::GaussLobatto;
    auto basis = QuadratureRule::quadrature_rule(n_colloc, basis_type);

    // construct Mesh
    const int nx = 20, ny = 20;
    Mesh2D mesh = Mesh2D::uniform_rect(nx, 0.0, 10.0, ny, -6.0, 6.0);
    
    // mesh statistics
    const int n_elem = mesh.n_elem(); // elements
    const int n_interior_faces = mesh.n_edges(FaceType::INTERIOR); 
    const int n_boundary_faces = mesh.n_edges(FaceType::BOUNDARY); 
    const int n_points = n_colloc * n_colloc * n_elem; // total number of collocation points
    const int n_dof = n_var * n_points; // number of degrees of freedom
    const double h = mesh.min_h(); // shortest length scale

    // Mass Matrix
    MassMatrix<approx_quad> m(mesh, basis); // m*u -> (u, v)

    // DG discretization
    DivF2D<approx_quad> div(mesh, basis);
    EdgeFluxF2D<approx_quad> interior_flux(mesh, FaceType::INTERIOR, basis);
    EdgeFluxF2D<approx_quad> boundary_flux(mesh, FaceType::BOUNDARY, basis);

    // map element DOFs to face values (for computing fluxes)
    auto interior_prol = make_face_prolongator(mesh, basis, FaceType::INTERIOR);
    Tensor<4,double> uI(n_colloc, n_var, 2, n_interior_faces); // face DOFs for interior faces

    auto boundary_prol = make_face_prolongator(mesh, basis, FaceType::BOUNDARY);
    Tensor<4,double> uB(n_colloc, n_var, 2, n_boundary_faces);

    // time interval: [0, T]
    double t = 0.0; // time variable
    const double T = 2.0;

    const double CFL = 1.0 / pow(n_colloc, 2); // Courant-Friedrich-Levy constant

    // this dt is optimal for forward Euler, for higher order we can typically take larger dt
    double dt = CFL / max_speed() * h ;
    const int nt = std::ceil(T / dt);
    dt = T / nt;

    std::cout << "#elements: " << n_elem << "\n"
              << "#DOFs/element: " << n_colloc << "^2\n"
              << "#DOFs: " << n_dof << "\n"
              << "dx: " << h << "\n"
              << "dt: " << dt << "\n"
              << "#times steps: " << nt << "\n";
    if (approx_quad)
        std::cout << "quadrature rule: fast (approximate)\n";
    else
        std::cout << "quadrature rule: exact\n";

    // du/dt = m \ [(f, grad v) - <f*, v>]
    auto time_derivative = [&](double * dudt, const double t, const double * u) -> void
    {
        for (int i = 0; i < n_dof; ++i)
            dudt[i] = 0.0;

        div.action(n_var, F, u, dudt);
        
        interior_prol->action(n_var, u, uI);
        interior_flux.action(n_var, numerical_flux, uI, uI);
        for (int i = 0; i < 2 * n_var * n_colloc * n_interior_faces; ++i)
        {
            uI[i] *= -1;
        }
        interior_prol->t(n_var, uI, dudt);

        boundary_prol->action(n_var, u, uB);
        
        // reflect interior values to exterior values
        for (int e = 0; e < n_boundary_faces; ++e)
        {
            for (int d = 0; d < n_var; ++d)
            {
                for (int i = 0; i < n_colloc; ++i)
                {
                    uB(i, d, 1, e) = uB(i, d, 0, e);
                }
            }
        }


        boundary_flux.action(n_var, numerical_flux, uB, uB);
        for (int i = 0; i < 2 * n_var * n_colloc * n_boundary_faces; ++i)
        {
            uB[i] *= -1;
        }
        boundary_prol->t(n_var, uB, dudt);

        m.inv(n_var, dudt);
    };

    // time integrator
    ode::SSPRK3 rk(n_dof);

    // set up solution vector
    Tensor<4, double> u(n_var, n_colloc, n_colloc, n_elem);

    // Project initial conditions
    LinearFunctional2D LF(mesh, basis);
    LF.action(n_var, initial_conditions, u);
    m.inv(n_var, u);

    // save solution collocation points to file
    auto x = mesh.element_metrics(basis).physical_coordinates();
    to_file("solution/x.00000", 2 * n_points, x);

    // save initial conditions to file
    to_file(std::format("solution/u{:0>5d}.00000", 0), n_dof, u);

    // Time loop
    ProgressBar progress_bar(nt);
    constexpr int skip = 10; // save solution every skip time steps
    for (int it = 1; it <= nt; ++it)
    {
        rk.step(dt, time_derivative, t, u);

        if (it % skip == 0)
            to_file(std::format("solution/u{:0>5d}.00000", it/skip), n_dof, u);

        ++progress_bar;
        std::cout << "[" << progress_bar.get() << "]" << std::setw(5) << it << " / " << nt << "\r" << std::flush;
    }
    std::cout << std::endl;

    return 0;
}