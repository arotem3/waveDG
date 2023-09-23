#include "WaveEquation.hpp"

namespace dg
{
    Advection::Advection(const Mesh2D& mesh, const QuadratureRule * basis)
        : ProjB(mesh, basis, BOUNDARY),
          ProjI(mesh, basis, INTERIOR)
    {
        auto c = [](const double x[2], double F[]) -> void
        {
            F[0] = -1.0;
            F[1] = 0.5;
        };

        double cV[2];
        double _zero[] = {0.0, 0.0};
        c(_zero, cV);

        const int neB = mesh.n_edges(BOUNDARY);
        const int neI = mesh.n_edges(INTERIOR);
        const int n_colloc = basis->n;

        dcube cB(n_colloc, 2, neB);
        dcube cI(n_colloc, 2, neI);

        auto quad = quadrature_rule(n_colloc+2);

        ProjB.project_normal(cV, cB.data(), true);
        ProjI.project_normal(cV, cI.data(), true);
        for (int e=0; e < neB; ++e)
        {
            for (int j=0; j < n_colloc; ++j)
            {
                cB(j, 1, e) = cB(j, 0, e);
            }
        }

        div.reset(new Div(1, mesh, basis, quad, cV, true));

        // in flux set b = 0.5 (instead of -0.5) because we are solving
        // du/dt + c.grad(u) = 0
        // -> du/dt = -div(cu)
        // -> (du/dt, v) = (c u, grad v) + a < n.c {u}, v > - b < n.c [u], v >
        // where as EdgeFlux discretizes a < n.c {u}, v> + b < n.c [u], v >
        FlxB.reset(new EdgeFlux(1, mesh, BOUNDARY, basis, quad, cB.data(), 1.0, 0.5));
        FlxI.reset(new EdgeFlux(1, mesh, INTERIOR, basis, quad, cI.data(), 1.0, 0.5));

        uB = dcube(n_colloc, 2, neB);
        uI = dcube(n_colloc, 2, neI);
    }

    void Advection::operator()(const double * x, double * y) const
    {
        // volume integral
        (*div)(x, y);

        // edge integrals
        ProjB(x, uB.data());
        (*FlxB)(uB.data());
        ProjB.t(uB.data(), y);

        ProjI(x, uI.data());
        (*FlxI)(uI.data());
        ProjI.t(uI.data(), y);
    }
} // namespace dg
