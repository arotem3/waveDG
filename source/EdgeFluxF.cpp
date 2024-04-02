#include "EdgeFluxF.hpp"

namespace dg
{
    EdgeFluxF1D::EdgeFluxF1D(const Mesh1D& mesh, FaceType face_type_, const QuadratureRule * basis)
        : face_type(face_type_),
          n_faces(mesh.n_faces(face_type_)),
          x(n_faces)
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

    template <>
    EdgeFluxF2D<true>::EdgeFluxF2D(const Mesh2D& mesh, FaceType face_type_, const QuadratureRule * basis, const QuadratureRule * quad)
        : face_type{face_type_},
          n_colloc{basis->n},
          n_faces{mesh.n_edges(face_type)},
          quad{basis},
          n_quad{basis->n}
    {
        auto& metrics = mesh.edge_metrics(basis, face_type);
        X       = reshape(metrics.physical_coordinates(), 2, n_colloc, n_faces);
        normals = reshape(metrics.normals(),              2, n_colloc, n_faces);
        meas    = reshape(metrics.measures(),                n_colloc, n_faces);
    }

    template <>
    EdgeFluxF2D<false>::EdgeFluxF2D(const Mesh2D& mesh, FaceType face_type_, const QuadratureRule * basis, const QuadratureRule * quad_)
        : face_type{face_type_},
          n_colloc{basis->n},
          n_faces{mesh.n_edges(face_type)},
          quad{quad_ ? quad_ : QuadratureRule::quadrature_rule(2 * n_colloc)},
          n_quad{quad->n},
          P(n_quad, n_colloc),
          Pt(n_colloc, n_quad)
    {
        auto& metrics = mesh.edge_metrics(quad, face_type);
        X       = reshape(metrics.physical_coordinates(), 2, n_quad, n_faces);
        normals = reshape(metrics.normals(),              2, n_quad, n_faces);
        meas    = reshape(metrics.measures(),                n_quad, n_faces);

        lagrange_basis(P, n_colloc, basis->x, n_quad, quad->x);
        for (int i = 0; i < n_quad; ++i)
        {
            for (int j = 0; j < n_colloc; ++j)
            {
                Pt(j, i) = P(i, j);
            }
        }
    }
} // namespace dg
