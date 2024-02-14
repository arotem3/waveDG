#include "PeriodicBC.hpp"
#include <iomanip>

namespace dg
{
    PeriodicBC1d::PeriodicBC1d(int nvar, const Mesh1D& mesh, const QuadratureRule * basis, const double * a, bool constant_coefficient)
        : n_var(nvar)
    {
        const int neB = mesh.n_faces(FaceType::BOUNDARY);
        const int v2d = n_var * n_var;

        if (neB == 0)
            return;

        face_prol = make_face_prolongator(v2d, mesh, basis, FaceType::BOUNDARY);

        dvec aB;
        if (constant_coefficient)
        {
            aB.reshape(v2d);

            for (int i = 0; i < v2d; ++i)
            {
                aB(i) = a[i];
            }
        }
        else
        {
            auto f = make_face_prolongator(v2d, mesh, basis, FaceType::BOUNDARY);

            aB.reshape(2 * v2d * neB);

            f->action(a, aB);
        }

        flx.reset(new EdgeFlux<false>(n_var, mesh, FaceType::BOUNDARY, basis, aB, constant_coefficient, -1.0, -0.5));

        uB.reshape(2 * n_var * neB);
    }

    void PeriodicBC1d::action(const double * u, double * F) const
    {
        face_prol->action(u, uB);

        // in serial periodic BC is simple
        auto uf = reshape(uB, n_var, 2, 2);
        for (int d = 0; d < n_var; ++d)
        {
            uf(d, 0, 0) = uf(d, 0, 1);
            uf(d, 1, 1) = uf(d, 1, 0);
        }
        
        flx->action(uB, uB);
        face_prol->t(uB, F);
    }
} // namespace dg
