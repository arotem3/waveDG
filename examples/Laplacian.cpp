#include "wavedg.hpp"
#include <iomanip>

using namespace dg;

void f(const double* x, double *z)
{
    double x2 = x[0] * x[0];
    double x4 = x2 * x2;
    *z = x[0] * x[1] * (x4 - 5.0) * (x[1] * x[1] - 3.0) / 24.0;
}

void Lf(const double* x, double *z)
{
    double x2 = x[0] * x[0];
    double x4 = x2 * x2;
    double a = x4 - 5;
    double b = x2 * (x[1] * x[1] - 3.0);
    *z = x[0] * x[1] * (0.25 * a + 5.0/6.0 * b);
}

int main()
{
    const int n_colloc = 7;

    auto basis = QuadratureRule::quadrature_rule(n_colloc);

    const int nx = 1, ny = 1;
    Mesh2D mesh = Mesh2D::uniform_rect(nx, -1.0, 1.0, ny, -1.0, 1.0);

    MassMatrix<false> m(mesh, basis);

    LinearFunctional2D F(mesh, basis);
    FEMVector u(1, mesh, basis);
    F.action(1, f, u);
    m.inv(1, u);

    FEMVector Lu(1, mesh, basis);
    F.action(1, Lf, Lu);
    m.inv(1, Lu);

    auto quad = QuadratureRule::quadrature_rule(10);
    Laplacian<false> L(mesh, basis, quad);
    FEMVector Au(1, mesh, basis);
    L.action(1, u, Au);
    m.inv(1, Au);

    const int n_dof = u.size();

    double err = 0.0;
    for (int i = 0; i < n_dof; ++i)
    {
        double e = Lu.get()[i] + Au.get()[i];
        err = std::max(err, std::abs(e));
    }

    std::cout << "max error : " << err << "\n";

    return 0;
}