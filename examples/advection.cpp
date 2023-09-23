#include <iostream>
#include <fstream>
#include <iomanip>

#include "WaveEquation.hpp"
#include "ode.hpp"

using namespace dg;

static Mesh2D load_mesh(const std::string& folder)
{
    std::ifstream info(folder + "/info.txt");
    if (not info)
        throw std::runtime_error("cannot open file: " + folder + "/info.txt");

    int n_pts, n_elem;
    info >> n_pts >> n_elem;
    info.close();

    dmat x(2, n_pts);
    Matrix<int> elems(4, n_elem);

    std::ifstream coo(folder + "/coordinates.txt");
    if (not coo)
        throw std::runtime_error("cannot open file: " + folder + "/coordinates.txt");

    for (int i = 0; i < n_pts; ++i)
    {
        coo >> x(0, i) >> x(1, i);
    }
    coo.close();

    std::ifstream elements(folder + "/elements.txt");
    if (not coo)
        throw std::runtime_error("cannot open file: " + folder + "/elements.txt");

    for (int i = 0; i < n_elem; ++i)
    {
        elements >> elems(0, i) >> elems(1, i) >> elems(2, i) >> elems(3, i);
    }
    elements.close();

    return Mesh2D::from_vertices(n_pts, x.data(), n_elem, elems.data());
}

class Callback
{
private:
    int it;
    u_long n;

public:
    Callback(u_long npts) : it(0), n(npts) {}

    inline bool operator()(double t, const double * xy, const double * v)
    {
        if (it == 0)
        {
            std::ofstream out("solution/x.dat", std::ios::out | std::ios::binary);
            out.write(reinterpret_cast<const char*>(xy), 2 * n * sizeof(double));
            out.close();
        }

        if (it % 10 == 0)
        {
            std::ofstream out("solution/u" + std::to_string(it) + ".dat", std::ios::out | std::ios::binary);
            out.write(reinterpret_cast<const char*>(v), n * sizeof(double));
            out.close();
        }

        ++it;

        return true;
    }
};

inline void initial_conditions(const double x[2], double F[])
{
    *F = std::sin(M_PI*x[0]) * std::sin(M_PI*x[1]);
}

int main()
{
    Mesh2D mesh = load_mesh("meshes/Square");
    const int n_elem = mesh.n_elem();

    const int n_colloc = 4;
    auto basis = quadrature_rule(n_colloc);//, QuadratureType::GaussLobatto);
    auto quad = quadrature_rule(n_colloc+2);

    const int n_dof = n_colloc * n_colloc * n_elem;

    Advection advec(mesh, basis);
    MassMatrix<false> m(mesh, basis, quad);

    auto time_derivative = [&](double * dudt, const double t, const double * u) -> void
    {
        advec(u, dudt);
        m.inv(dudt);

        for (int i = 0; i < n_dof; ++i)
            dudt[i] *= -1.0;
    };

    dcube u(n_colloc, n_colloc, n_elem);

    // initial conditions
    Projector Proj(mesh, m, basis, quad);
    Proj(initial_conditions, u.data());

    auto x = mesh.element_physical_coordinates(basis);

    const double T = 1.0;
    double dt = 0.001;
    const int nt = std::ceil(T / dt);
    dt = T / nt;

    ode::RungeKutta2 rk(n_dof);
    double t = 0.0;

    Callback callback(n_dof);
    bool check = callback(t, x, u.data());
    std::string progress(30, ' ');
    for (int it = 1; check && it <= nt; ++it)
    {
        rk.step(dt, time_derivative, t, u.data());

        check = callback(t, x, u.data());

        progress.at(30*(it-1)/nt) = '#';
        std::cout << "[" << progress << "]" << std::setw(5) << it << " / " << nt << "\r" << std::flush;
    }
    std::cout << "\n";

    return 0;
}