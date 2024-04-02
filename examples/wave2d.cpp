/** @file wave2d.cpp
 *  @brief Example driver for solving the wave equation.
 * 
 * This file is a driver for solving the wave equation in first order form:
 * $$p_t + \nabla \cdot \mathbf{u} = f(t, x),$$
 * $$\mathbf{u}_t + \nabla p = \mathbf{g}(t, x).$$
 * 
 * When \f$\mathbf{g}(t, x) \equiv 0\f$ we can elliminate \f$\mathbf{u}\f$ to find that \f$p\f$ satisfies the classical wave equation:
 * $$p_{tt} = \Delta p - \partial_t f(t, x).$$
 * 
 * To compile & run this program, first compile the library in serial mode:
 * 
 * `cmake . -D WDG_USE_MPI=OFF`
 * `make wavedg -j`
 * 
 * Then compile this file with
 * 
 * `make wave2d`
 * 
 * And run with
 * 
 * `./examples/wave2d`
 * 
 * Adjusting the number of processors as needed.
 * The program will write the collocation points and solution
 * values to `solution/x.00000` and `solution/u%05d.00000`, respectively, in binary format.
 * Where the number on u is the time step; e.g. first processor writes:
 * `solution/x.00000` and at time step 10 will write `solution/u00010.00000`, etc.
 */

#include <iostream>
#include <iomanip>
#include <format>

#include "wavedg.hpp"

using namespace dg;

/// saves solution vector to binary file.
inline static void to_file(const std::string& fname, int n_dof, const double * u)
{
    std::ofstream out(fname, std::ios::out | std::ios::binary);
    out.write(reinterpret_cast<const char*>(u), n_dof * sizeof(double));
    out.close();
}

/// specify initial condition.
inline static void initial_conditions(const double x[2], double F[])
{
    F[0] = 0.0;
    F[1] = 0.0;
    F[2] = 0.0;
}

/// the forcing terms F = (f(t, x), g(t, x))
inline static void force(const double t, const double x[2], double F[])
{
    const double r = x[0] * x[0] + x[1] * x[1];
    F[0] = 10.0 * std::exp(-100.0 * r) * std::sin(30 * t);
    F[1] = 0.0;
    F[2] = 0.0;
}

/// @brief specifies the boundary conditions for every edge in `mesh` assuming a the geometry: [-1, 1]x[-1, 1].
/// @param mesh the mesh representing the geometry [-1, 1]x[1, 1].
/// @return integer vector where out[k] is the boundary condition for boundary
/// edge k. If out[k] == 0, then edge k is an approximate absorbing
/// (non-reflecting) boundary, if out[k] == 1, then edge k is a Neumann
/// (reflecting) boundary.
static ivec boundary_conditions(const Mesh2D& mesh)
{
    const int nB = mesh.n_edges(FaceType::BOUNDARY);
    constexpr int REFLECT = 1;
    constexpr int ABSORB = 0;

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
            bc(e) = REFLECT;
        else
            bc(e) = ABSORB;
    }

    return bc;
}

static constexpr double max_speed()
{
    return 1.0;
}

int main(int argc, char ** argv)
{
    // approx_quad == true ==> compute integrals using quadrature rule corresponding to the Lagrange basis collocation points.
    // approx_quad == false ==> compute integrals on higher order quadrature rule (automatically determined).
    constexpr bool approx_quad = true;

    // vector dimension of PDE. In this case 3: (p, u[0], u[1]). We can
    // interpret p as the pressure field and u=(u[0], u[1]) as the velocity (or
    // momentum) field.
    constexpr int n_var = 3;

    // Specify basis functions in terms of 1D quadrature rule. Basis functions
    // are tensor product of 1D Lagrange interpolating polynomials on Gauss
    // quadrature rule. The number of collocation points for 1D quadrature rule.
    // The order of the DG discretization is n_colloc - 1/2.
    const int n_colloc = 6;
    QuadratureRule::QuadratureType basis_type = QuadratureRule::GaussLobatto;
    auto basis = QuadratureRule::quadrature_rule(n_colloc, basis_type);

    // construct Mesh
    const int nx = 20, ny = 20;
    const double x_min = -1.0, x_max = 1.0, y_min = -1.0, y_max = 1.0;
    Mesh2D mesh = Mesh2D::uniform_rect(nx, x_min, x_max, ny, y_min, y_max);
    
    // mesh statistics
    const int n_elem = mesh.n_elem(); // elements on processor
    const int n_points = n_colloc * n_colloc * n_elem; // local total number of collocation points
    const int n_dof = n_var * n_points; // local number of degrees of freedom
    const double h = mesh.min_h(); // shortest length scale

    // time interval: [0, T]
    double t = 0.0; // time variable
    const double T = 2.0;

    const double CFL = 1.0 / std::pow(n_colloc, 2); // Courant-Friedrich-Levy constant

    // this dt is optimal for forward Euler, for higher order we can take larger dt
    double dt = CFL / max_speed() * h;
    const int nt = std::ceil(T / dt);
    dt = T / nt;

    std::cout << "#elements: " << n_elem << "\n"
                << "#DOFs/element: " << n_var << " x " << n_colloc << "^2\n"
                << "#DOFs: " << n_dof << "\n"
                << "dx: " << h << "\n"
                << "dt: " << dt << "\n"
                << "#times steps: " << nt << "\n";
    if (approx_quad)
        std::cout << "quadrature rule: fast (approximate)\n";
    else
        std::cout << "quadrature rule: exact\n";

    // DG discretization:
    WaveEquation a(mesh, basis, approx_quad);
    MassMatrix<approx_quad> m(mesh, basis);
    
    // Boundary conditions
    const ivec _bc = boundary_conditions(mesh);
    WaveBC bc(mesh, _bc, basis, approx_quad);

    // Forcing term
    LinearFunctional2D L(mesh, basis);
    FEMVector f(n_var, mesh, basis);
    auto F = reshape(f.get(), f.size());

    // m * du/dt = a*u + bc*u + f -> du/dt = m \ (a * u + bc * u + f).
    auto time_derivative = [&](double * dudt, const double t, const double * u) -> void
    {
        for (int i=0; i < n_dof; ++i)
            dudt[i] = 0.0;
            
        a.action(u, dudt);
        bc.action(u, dudt);

        L.action(n_var, [t](const double * x_, double * f_) -> void {force(t,x_,f_);}, F);
        for (int i=0; i < n_dof; ++i)
            dudt[i] += F(i);

        m.inv(n_var, dudt);
    };

    // time integrator
    ode::RungeKutta4 rk(n_dof);

    // set up solution vector.
    FEMVector u(n_var, mesh, basis);

    // initial conditions
    L.action(n_var, initial_conditions, u);
    m.inv(n_var, u);

    // save solution collocation points to file
    auto x = mesh.element_metrics(basis).physical_coordinates();
    to_file("solution/x.00000", 2*n_points, x);
    
    // save initial condition to file
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