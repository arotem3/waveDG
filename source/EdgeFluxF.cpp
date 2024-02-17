#include "EdgeFluxF.hpp"

namespace dg
{
    EdgeFluxF1D::EdgeFluxF1D(int n_var_, const Mesh1D& mesh, FaceType face_type_, const QuadratureRule * basis)
        : face_type(face_type_),
          n_var(n_var_),
          n_faces(mesh.n_faces(face_type_)),
          x(n_faces),
          fh(n_var),
          uL(n_var),
          uR(n_var)
    {
        for (int f = 0; f < n_faces; ++f)
        {
            auto& face = mesh.face(f, face_type);
            const int s = (face.elements[0] >= 0) ? 0 : 1;
            const int el = face.elements[s];

            auto& elem = mesh.element(el);
            x(f) = elem.end_points()[1-s];
        }
    }
} // namespace dg
