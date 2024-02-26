/** @file cl1d.cpp
 *  @brief Example driver for solving the 1D invicid Burger's equation
 * 
 * This file is a driver for solving the 1D invicid Burger's equation:
 * $$u_t + \left(\frac{1}{2}u^2\right)_x = 0.$$
 * 
 * To compile & run this program, first compile the library in serial mode:
 * 
 * `cmake . -D WDG_USE_MPI=OFF`
 * `make wavedg -j`
 * 
 * Then compile this file with
 * 
 * `make cl1d`
 * 
 * And run with
 * 
 * `./examples/cl1d`
 * 
 * The program will write the collocation points and solution
 * values to `solution/x.00000` and `solution/u%05d.00000`, respectively, in binary format.
 * Where the number on u is the time step; e.g. at time step 10 we write `solution/u00010.00000` 
 * and at time step 555 we write `solution/u00555.00000`, etc.
 */

#include <iostream>
#include <iomanip>
#include <fstream>
#include <format>

#include "wavedg.hpp"

using namespace dg;

inline static void initial_conditions(const double x[], double u[])
{
    *u = 1 + 0.5 * std::exp(-30.0 * x[0] * x[0]);
}

inline static void to_file(const std::string& fname, int n_dof, const double * u)
{
    std::ofstream out(fname, std::ios::out | std::ios::binary);
    out.write(reinterpret_cast<const char*>(u), n_dof * sizeof(double));
    out.close();
}

// invicid Burger's equation in conservative form
inline static void F(double x, const double u[1], double f[1])
{
    *f = 0.5 * u[0] * u[0];
}

// local Lax-Friedrichs Flux
inline static void LF_flux(double x, const double uL[], const double uR[], double fh[])
{
    double fL, fR;
    F(x, uL, &fL);
    F(x, uR, &fR);

    const double avg = 0.5 * (fL + fR);
    const double c = std::max(std::abs(uL[0]), std::abs(uR[0]));
    const double jmp = uL[0] - uR[0];

    fh[0] = avg + 0.5 * c * jmp;
}

constexpr static double max_speed()
{
    return 1.5;
}

int main(int argc, char ** argv)
{
    // approx_quad == true ==> compute integrals using quadrature rule corresponding to the Lagrange basis collocation points.
    // approx_quad == false ==> compute integrals on higher order quadrature rule (automatically determined).
    constexpr bool approx_quad = true;

    // Specify basis functions in terms of quadrature rule. Basis functions are
    // the Lagrange interpolating polynomials on Gauss quadrature rule. The
    // order of the DG discretization is n_colloc - 1/2.
    const int n_colloc = 3;
    QuadratureRule::QuadratureType basis_type = QuadratureRule::GaussLegendre;
    auto basis = QuadratureRule::quadrature_rule(n_colloc, basis_type);

    // construct Mesh.
    const int n_elem = 40;
    const bool periodic = true;
    Mesh1D mesh = Mesh1D::uniform_mesh(n_elem, -1.0, 1.0, periodic);

    // mesh statistics
    const int n_points = n_colloc * n_elem;
    const int n_dof = n_points;
    const double h = mesh.min_h();

    // Mass Matrix
    MassMatrix<approx_quad> m(1, mesh, basis);

    // time interval: [0, T]
    double t = 0.0; // time variable
    const double T = 2.0;
    
    const double CFL = 1.0 / std::pow(n_colloc, 2); // Courant-Friedrich-Levy constant

    double dt = CFL / max_speed() * h;
    const int nt = std::ceil(T / dt);
    dt = T / nt;

    std::cout << "#elements: " << n_elem << "\n"
              << "#DOFs/element: " << n_colloc << "\n"
              << "#DOFs: " << n_dof << "\n"
              << "dx: " << h << "\n"
              << "dt: " << dt << "\n"
              << "#times steps: " << nt << "\n";
    if (approx_quad)
        std::cout << "quadrature rule: fast (approximate)\n";
    else
        std::cout << "quadrature rule: exact\n";

    // DG discretization
    DivF1D<approx_quad> div(1, mesh, basis);
    EdgeFluxF1D flx(1, mesh, FaceType::INTERIOR, basis);

    // map element DOFs to face values (for computing fluxes)
    auto prolongator = make_face_prolongator(1, mesh, basis, FaceType::INTERIOR);
    const int n_faces = mesh.n_faces(FaceType::INTERIOR);
    dmat uI(2, n_faces); // face DOFs for interior faces

    // m * du/dt = (f, v_x) - <f*, v>.
    auto time_derivative = [&](double * dudt, const double t, const double * u) -> void
    {
        for (int i = 0; i < n_dof; ++i)
            dudt[i] = 0.0;

        div.action(F, u, dudt);

        prolongator->action(u, uI); // compute face values
        flx.action(LF_flux, uI, uI); // compute flux inplace on uI
        
        for (int i = 0; i < n_faces; ++i)
        {
            uI(0, i) *= -1;
            uI(1, i) *= -1;
        }

        prolongator->t(uI, dudt); // add flux to dudt

        m.inv(dudt);
    };

    // time integrator
    ode::SSPRK3 rk(n_dof);

    // set up solution vector
    dmat u(n_colloc, n_elem);

    // Project initial conditions
    LinearFunctional LF(1, mesh, basis);
    LF(initial_conditions, u);
    m.inv(u);

    // save solution collocation points to file
    auto x = mesh.element_metrics(basis).physical_coordinates();
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