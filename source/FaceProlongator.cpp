#include "FaceProlongator.hpp"

namespace dg
{
    FaceProlongator::FaceProlongator(const Mesh2D& mesh, const QuadratureRule * basis, FaceType edge_type_)
        : dim(2),
          n_elem(mesh.n_elem()),
          n_edges(mesh.n_edges(edge_type_)),
          n_colloc(basis->n),
          face_type(edge_type_)
          #ifdef WDG_USE_MPI
          ,lfp{mesh.face_pattern(face_type)}
          #endif
    {}

    FaceProlongator::FaceProlongator(const Mesh1D& mesh, const QuadratureRule * basis, FaceType edge_type_)
        : dim(1),
          n_elem(mesh.n_elem()),
          n_edges(mesh.n_faces(edge_type_)),
          n_colloc(basis->n),
          face_type(edge_type_)
          #ifdef WDG_USE_MPI
          ,lfp{mesh.face_pattern(face_type)}
          #endif
    {}

#ifdef WDG_USE_MPI
    LobattoFaceProlongator::LobattoFaceProlongator(const Mesh2D& mesh, const QuadratureRule * basis, FaceType edge_type_)
        : FaceProlongator(mesh, basis, edge_type_)
    {
        if (n_edges == 0)
            return;

        _v2e.reshape(2 * n_colloc * n_edges);
        _v2e.fill(-1);
        auto v2e = reshape(_v2e, n_colloc, 2, n_edges);

        const int nc = n_colloc;
        auto mapV2E = [nc,&mesh](int i, int f, int el) -> int
        {
            el = mesh.local_element_index(el);
            const int m = (f == 0 || f == 2) ? i : (f == 1) ? (nc-1) : 0;
            const int n = (f == 1 || f == 3) ? i : (f == 2) ? (nc-1) : 0;

            return m + nc * (n + nc * el);
        };

        for (int l : lfp)
        {
            const int s = l % 2;
            const int e = l / 2;

            auto const edge = mesh.edge(e, face_type);

            const int el = edge->elements[s];
            const int f = edge->sides[s];
            const bool reversed = (s == 1) && (edge->delta < 0); // whether the degrees of freedom of the second element need to be reversed to match the first element.
            const int inc = (reversed) ? -1 : 1;

            int j = (reversed) ? (n_colloc - 1) : 0;
            for (int i=0; i < n_colloc; ++i)
            {
                v2e(i, s, e) = mapV2E(j, f, el);
                j += inc;
            }
        }
    }

    LobattoFaceProlongator::LobattoFaceProlongator(const Mesh1D& mesh, const QuadratureRule * basis, FaceType edge_type_)
        : FaceProlongator(mesh, basis, edge_type_)
    {
        if (n_edges == 0)
            return;

        _v2e.reshape(2 * n_edges);
        _v2e.fill(-1);
        auto v2e = reshape(_v2e, 2, n_edges);

        for (int l : lfp)
        {
            const int s = l % 2;
            const int e = l / 2;

            auto& face = mesh.face(e, face_type);

            const int el = mesh.local_element_index(face.elements[s]);
            
            const int idx = (s == 0) ? (n_colloc-1) : 0;
            v2e(s, e) = idx + n_colloc * el;
        }
    }

    static void lobatto_action_2d(int n_elem, int n_edges, int n_colloc, int n_var, const double * u_, double * uf_, const_ivec_wrapper local_face_pattern, const ivec& _v2e)
    {
        auto uf = reshape(uf_, n_colloc, n_var, 2, n_edges);
        auto u = reshape(u_, n_var, n_colloc * n_colloc * n_elem);

        auto v2e = reshape(_v2e, n_colloc, 2, n_edges);

        uf.zeros();

        // prolongate all face values from local elements
        for (int l : local_face_pattern)
        {
            const int s = l % 2;
            const int e = l / 2;

            for (int i = 0; i < n_colloc; ++i)
            {
                const int j = v2e(i, s, e);

                for (int d = 0; d < n_var; ++d)
                {
                    uf(i, d, s, e) = u(d, j);
                }
            }
        }
    }

    static void lobatto_action_1d_interior(int n_elem, int n_edges, int n_colloc, int n_var, const double * u_, double * uf_, const_ivec_wrapper local_face_pattern, const ivec& _v2e)
    {
        auto uf = reshape(uf_, n_var, 2, n_edges);
        auto u = reshape(u_, n_var, n_colloc * n_elem);

        auto v2e = reshape(_v2e, 2, n_edges);

        for (int l : local_face_pattern)
        {
            const int s = l % 2;
            const int e = l / 2;

            const int j  = v2e(s, e);
            for (int d = 0; d < n_var; ++d)
            {
                uf(d, s, e) = u(d, j);
            }
        }
    }

    static void lobatto_action_1d_boundary(int n_elem, int n_edges, int n_colloc, int n_var, const double * u_, double * uf_, const_ivec_wrapper local_face_pattern, const ivec& _v2e)
    {
        auto uf = reshape(uf_, n_var, 2, n_edges);
        auto u = reshape(u_, n_var, n_colloc * n_elem);

        auto v2e = reshape(_v2e, 2, n_edges);

        for (int l : local_face_pattern)
        {
            const int s = l % 2;
            const int e = l / 2;

            const int j  = v2e(s, e);
            for (int d = 0; d < n_var; ++d)
            {
                uf(d, 0, e) = u(d, j);
                uf(d, 1, e) = u(d, j);
            }
        }
    }

    void LobattoFaceProlongator::action(int n_var, const double * u_, double * uf_) const
    {
        if (n_edges == 0)
            return;

        switch (dim)
        {
        case 1:
            if (face_type == FaceType::INTERIOR)
                lobatto_action_1d_interior(n_elem, n_edges, n_colloc, n_var, u_, uf_, lfp, _v2e);
            else
                lobatto_action_1d_boundary(n_edges, n_edges, n_colloc, n_var, u_, uf_, lfp, _v2e);
            break;
        case 2:
            lobatto_action_2d(n_elem, n_edges, n_colloc, n_var, u_, uf_, lfp, _v2e);
            break;
        default:
            break;
        }
    }

    static void lobatto_transpose_2d(int n_elem, int n_edges, int n_colloc, int n_var, const double * uf_, double * u_, const_ivec_wrapper local_face_pattern, const ivec& _v2e)
    {
        auto uf = reshape(uf_, n_colloc, n_var, 2, n_edges);
        auto u = reshape(u_, n_var, n_colloc * n_colloc * n_elem);

        auto v2e = reshape(_v2e, n_colloc, 2, n_edges);

        for (int l : local_face_pattern)
        {
            const int s = l % 2;
            const int e = l / 2;

            for (int i = 0; i < n_colloc; ++i)
            {
                const int j = v2e(i, s, e);

                for (int d = 0; d < n_var; ++d)
                {
                    u(d, j) += uf(i, d, s, e);
                }
            }
        }
    }

    static void lobatto_transpose_1d(int n_elem, int n_edges, int n_colloc, int n_var, const double * uf_, double * u_, const_ivec_wrapper local_face_pattern, const ivec& _v2e)
    {
        auto uf = reshape(uf_, n_var, 2, n_edges);
        auto u = reshape(u_, n_var, n_colloc * n_elem);

        auto v2e = reshape(_v2e, 2, n_edges);

        for (int l : local_face_pattern)
        {
            const int s = l % 2;
            const int e = l / 2;

            const int j = v2e(s, e);

            for (int d = 0; d < n_var; ++d)
                u(d, j) += uf(d, s, e);
        }
    }

    void LobattoFaceProlongator::t(int n_var, const double * uf_, double * u_) const
    {
        if (n_edges == 0)
            return;
        
        switch (dim)
        {
        case 1:
            lobatto_transpose_1d(n_elem, n_edges, n_colloc, n_var, uf_, u_, lfp, _v2e);
            break;
        case 2:
            lobatto_transpose_2d(n_elem, n_edges, n_colloc, n_var, uf_, u_, lfp, _v2e);
            break;
        default:
            break;
        }
    }

    LegendreFaceProlongator::LegendreFaceProlongator(const Mesh2D& mesh, const QuadratureRule * basis, FaceType edge_type_)
        : FaceProlongator(mesh, basis, edge_type_),
          P(n_colloc)
    {
        if (n_edges == 0)
            return;

        constexpr double x[] = {-1.0};
        lagrange_basis(P, n_colloc, basis->x, 1, x);

        _v2e.reshape(2 * n_colloc * n_colloc * n_edges);
        _v2e.fill(-1);
        auto v2e = reshape(_v2e, n_colloc, n_colloc, 2, n_edges);

        const int nc = n_colloc;
        auto mapV2E = [nc,&mesh](int k, int i, int f, int el) -> int
        {
            el = mesh.local_element_index(el);
            const int m = (f == 0 || f == 2) ? i : (f == 1) ? (nc-1-k) : k;
            const int n = (f == 1 || f == 3) ? i : (f == 2) ? (nc-1-k) : k;

            return m + nc * (n + nc * el);
        };

        for (int l : lfp)
        {
            const int s = l % 2;
            const int e = l / 2;

            auto const edge = mesh.edge(e, face_type);

            const int el = edge->elements[s];
            const int f = edge->sides[s];
            const bool reversed = (s == 1) && (edge->delta < 0); // whether the degrees of freedom of the second element need to be reversed to match the first element.
            const int inc = (reversed) ? -1 : 1;

            int j = (reversed) ? (n_colloc - 1) : 0;
            for (int i=0; i < n_colloc; ++i)
            {
                for (int k=0; k < n_colloc; ++k)
                {
                    v2e(k, i, s, e) = mapV2E(k, j, f, el);
                }
                j += inc;
            }
        }
    }

    LegendreFaceProlongator::LegendreFaceProlongator(const Mesh1D& mesh, const QuadratureRule * basis, FaceType face_type)
        : FaceProlongator(mesh, basis, face_type),
          P(n_colloc)
    {
        if (n_edges == 0)
            return;

        constexpr double x[] = {-1.0};
        lagrange_basis(P, n_colloc, basis->x, 1, x);

        _v2e.reshape(2 * n_colloc * n_edges);
        _v2e.fill(-1);
        auto v2e = reshape(_v2e, n_colloc, 2, n_edges);

        for (int l : lfp)
        {
            const int s = l % 2;
            const int e = l / 2;

            auto& face = mesh.face(e, face_type);

            const int el = mesh.local_element_index(face.elements[s]);

            int j = (s == 0) ? (n_colloc-1) : 0;
            const int inc = (s == 0) ? -1 : 1;

            for (int i = 0; i < n_colloc; ++i)
            {
                v2e(i, s, e) = j + n_colloc * el;
                j += inc;
            }
        }
    }

    void legendre_action_2d(int n_elem, int n_edges, int n_colloc, int n_var, const double * u_, double * uf_, const_ivec_wrapper local_face_pattern, const ivec& _v2e, const dvec& P)
    {
        auto uf = reshape(uf_, n_colloc, n_var, 2, n_edges);
        auto u = reshape(u_, n_var, n_colloc * n_colloc * n_elem);

        auto v2e = reshape(_v2e, n_colloc, n_colloc, 2, n_edges);

        uf.zeros();

        for (int l : local_face_pattern)
        {
            const int s = l % 2;
            const int e = l / 2;

            for (int i = 0; i < n_colloc; ++i)
            {
                for (int d = 0; d < n_var; ++d)
                {
                    double val = 0.0;
                    for (int k = 0; k < n_colloc; ++k)
                    {
                        const int j = v2e(k, i, s, e);
                        val += u(d, j) * P(k);
                    }
                    uf(i, d, s, e) = val;
                }
            }
        }
    }

    void legendre_action_1d_interior(int n_elem, int n_edges, int n_colloc, int n_var, const double * u_, double * uf_, const_ivec_wrapper local_face_pattern, const ivec& _v2e, const dvec& P)
    {
        auto uf = reshape(uf_, n_var, 2, n_edges);
        auto u = reshape(u_, n_var, n_colloc * n_elem);

        auto v2e = reshape(_v2e, n_colloc, 2, n_edges);

        for (int l : local_face_pattern)
        {
            const int s = l % 2;
            const int e = l / 2;

            for (int d = 0; d < n_var; ++d)
            {
                double val = 0.0;
                for (int k = 0; k < n_colloc; ++k)
                {
                    const int j = v2e(k, s, e);
                    val += u(d, j) * P(k);
                }
                uf(d, s, e) = val;
            }
        }
    }

    void legendre_action_1d_boundary(int n_elem, int n_edges, int n_colloc, int n_var, const double * u_, double * uf_, const_ivec_wrapper local_face_pattern, const ivec& _v2e, const dvec& P)
    {
        auto uf = reshape(uf_, n_var, 2, n_edges);
        auto u = reshape(u_, n_var, n_colloc * n_elem);

        auto v2e = reshape(_v2e, n_colloc, 2, n_edges);

        for (int l : local_face_pattern)
        {
            const int s = l % 2;
            const int e = l / 2;

            for (int d = 0; d < n_var; ++d)
            {
                double val = 0.0;
                for (int k = 0; k < n_colloc; ++k)
                {
                    const int j = v2e(k, s, e);
                    val += u(d, j) * P(k);
                }
                uf(d, 0, e) = val;
                uf(d, 1, e) = val;
            }
        }
    }

    void LegendreFaceProlongator::action(int n_var, const double * u_, double * uf_) const
    {
        if (n_edges == 0)
            return;

        switch (dim)
        {
        case 1:
            if (face_type == FaceType::INTERIOR)
                legendre_action_1d_interior(n_elem, n_edges, n_colloc, n_var, u_, uf_, lfp, _v2e, P);
            else
                legendre_action_1d_boundary(n_elem, n_edges, n_colloc, n_var, u_, uf_, lfp, _v2e, P);
            break;
        case 2:
            legendre_action_2d(n_edges, n_edges, n_colloc, n_var, u_, uf_, lfp, _v2e, P);
            break;
        default:
            break;
        }
    }

    void legendre_transpose_2d(int n_elem, int n_edges, int n_colloc, int n_var, const double * uf_, double * u_, const_ivec_wrapper local_face_pattern, const ivec& _v2e, const dvec& P)
    {
        auto uf = reshape(uf_, n_colloc, n_var, 2, n_edges);
        auto u = reshape(u_, n_var, n_colloc * n_colloc * n_elem);

        auto v2e = reshape(_v2e, n_colloc, n_colloc, 2, n_edges);

        for (int l : local_face_pattern)
        {
            const int s = l % 2;
            const int e = l / 2;

            for (int i = 0; i < n_colloc; ++i)
            {
                for (int d = 0; d < n_var; ++d)
                {
                    double Y = uf(i, d, s, e);
                    for (int k = 0; k < n_colloc; ++k)
                    {
                        const int j = v2e(k, i, s, e);
                        u(d, j) += Y * P(k);
                    }
                }
            }
        }
    }

    void legendre_transpose_1d(int n_elem, int n_edges, int n_colloc, int n_var, const double * uf_, double * u_, const_ivec_wrapper local_face_pattern, const ivec& _v2e, const dvec& P)
    {
        auto uf = reshape(uf_, n_var, 2, n_edges);
        auto u = reshape(u_, n_var, n_colloc * n_elem);

        auto v2e = reshape(_v2e, n_colloc, 2, n_edges);

        for (int l : local_face_pattern)
        {
            const int s = l % 2;
            const int e = l / 2;

            for (int d = 0; d < n_var; ++d)
            {
                const double Y = uf(d, s, e);
                for (int k = 0; k < n_colloc; ++k)
                {
                    const int j = v2e(k, s, e);
                    u(d, j) += Y * P(k);
                }
            }
        }
    }

    void LegendreFaceProlongator::t(int n_var, const double * uf_, double * u_) const
    {
        if (n_edges == 0)
            return;
            
        switch (dim)
        {
        case 1:
            legendre_transpose_1d(n_elem, n_edges, n_colloc, n_var, uf_, u_, lfp, _v2e, P);
            break;
        case 2:
            legendre_transpose_2d(n_elem, n_edges, n_colloc, n_var, uf_, u_, lfp, _v2e, P);
            break;
        default:
            break;
        }
    }
#else
// LobattoFaceProlongator
    LobattoFaceProlongator::LobattoFaceProlongator(const Mesh2D& mesh, const QuadratureRule * basis, FaceType edge_type_)
        : FaceProlongator(mesh, basis, edge_type_)
    {
        if (n_edges == 0)
            return;

        _v2e.reshape(2 * n_colloc * n_edges);
        _v2e.fill(-1);
        auto v2e = reshape(_v2e, n_colloc, 2, n_edges);

        const int nc = n_colloc;
        auto mapV2E = [nc](int i, int f, int el) -> int
        {
            const int m = (f == 0 || f == 2) ? i : (f == 1) ? (nc-1) : 0;
            const int n = (f == 1 || f == 3) ? i : (f == 2) ? (nc-1) : 0;

            return m + nc * (n + nc * el);
        };

        for (int e = 0; e < n_edges; ++e)
        {
            const Edge * edge = mesh.edge(e, face_type);
            const int el0 = edge->elements[0];
            const int s0 = edge->sides[0];

            for (int i = 0; i < n_colloc; ++i)
            {
                v2e(i, 0, e) = mapV2E(i, s0, el0);
            }

            if (face_type == FaceType::INTERIOR)
            {
                const int el1 = edge->elements[1];
                const int s1 = edge->sides[1];

                for (int i = 0; i < n_colloc; ++i)
                {
                    const int j = (edge->delta > 0) ? i : (n_colloc - 1 - i);
                    v2e(i, 1, e) = mapV2E(j, s1, el1);
                }
            }
        }
    }

    LobattoFaceProlongator::LobattoFaceProlongator(const Mesh1D& mesh, const QuadratureRule * basis, FaceType edge_type_)
        : FaceProlongator(mesh, basis, edge_type_)
    {
        if (n_edges == 0)
            return;

        _v2e.reshape(2 * n_edges);
        _v2e.fill(-1);
        auto v2e = reshape(_v2e, 2, n_edges);
        
        for (int e = 0; e < n_edges; ++e)
        {
            auto& face = mesh.face(e, face_type);

            const int el0 = face.elements[0];
            if (el0 >= 0)
                v2e(0, e) = (n_colloc-1) + n_colloc * el0;

            const int el1 = face.elements[1];
            if (el1 >= 0)
                v2e(1, e) = (0) + n_colloc * el1;
        }
    }

    static void lobatto_action_2d(int n_elem, int n_edges, int n_colloc, int n_var, FaceType face_type, const double * u_, double * uf_, const int * v2e_)
    {
        auto uf = reshape(uf_, n_colloc, n_var, 2, n_edges);
        auto u = reshape(u_, n_var, n_colloc * n_colloc * n_elem);
        auto v2e = reshape(v2e_, n_colloc, 2, n_edges);

        const int n_sides = (face_type == FaceType::INTERIOR) ? 2 : 1;

        uf.zeros();

        for (int e = 0; e < n_edges; ++e)
        {
            for (int side = 0; side < n_sides; ++side)
            {
                for (int i = 0; i < n_colloc; ++i)
                {
                    const int j = v2e(i, side, e);

                    for (int d = 0; d < n_var; ++d)
                    {    
                        uf(i, d, side, e) = u(d, j);
                    }
                }
            }
        }
    }

    static void lobatto_action_1d_interior(int n_elem, int n_edges, int n_colloc, int n_var, const double * u_, double * uf_, const int * v2e_)
    {
        auto uf = reshape(uf_, n_var, 2, n_edges);
        auto u = reshape(u_, n_var, n_colloc * n_elem);
        auto v2e = reshape(v2e_, 2, n_edges);

        for (int e = 0; e < n_edges; ++e)
        {
            for (int side = 0; side < 2; ++side)
            {
                const int j = v2e(side, e);
                for (int d = 0; d < n_var; ++d)
                {
                    uf(d, side, e) = u(d, j);
                }
            }
        }
    }

    static void lobatto_action_1d_boundary(int n_elem, int n_edges, int n_colloc, int n_var, const double * u_, double * uf_, const int * v2e_)
    {
        auto uf = reshape(uf_, n_var, 2, n_edges);
        auto u = reshape(u_, n_var, n_colloc * n_elem);
        auto v2e = reshape(v2e_, 2, n_edges);

        for (int e = 0; e < n_edges; ++e)
        {
            const int j = (v2e(0, e) >= 0) ? v2e(0, e) : v2e(1, e); // only one is valid
            for (int d = 0; d < n_var; ++d)
            {
                uf(d, 0, e) = u(d, j);
                uf(d, 1, e) = u(d, j);
            }
        }
    }

    void LobattoFaceProlongator::action(int n_var, const double * u_, double * uf_) const
    {
        if (n_edges == 0)
            return;

        switch (dim)
        {
        case 1:
            if (face_type == FaceType::INTERIOR)
                lobatto_action_1d_interior(n_elem, n_edges, n_colloc, n_var, u_, uf_, _v2e);
            else
                lobatto_action_1d_boundary(n_elem, n_edges, n_colloc, n_var, u_, uf_, _v2e);
            break;
        case 2:
            lobatto_action_2d(n_elem, n_edges, n_colloc, n_var, face_type, u_, uf_, _v2e);
            break;
        default:
            wdg_error("LobattoFaceProlongator::action not implemented for dim == " + std::to_string(dim));
            break;
        }
    }

    static void lobatto_transpose_2d(int n_elem, int n_edges, int n_colloc, int n_var, FaceType face_type, const double * uf_, double * u_, const int * v2e_)
    {
        auto uf = reshape(uf_, n_colloc, n_var, 2, n_edges);
        auto u = reshape(u_, n_var, n_colloc * n_colloc * n_elem);
        auto v2e = reshape(v2e_, n_colloc, 2, n_edges);

        const int n_sides = (face_type == FaceType::INTERIOR) ? 2 : 1;

        for (int e = 0; e < n_edges; ++e)
        {
            for (int side = 0; side < n_sides; ++side)
            {
                for (int i = 0; i < n_colloc; ++i)
                {
                    for (int d = 0; d < n_var; ++d)
                    {
                        const int j = v2e(i, side, e);
                        u(d, j) += uf(i, d, side, e);
                    }
                }
            }
        }
    }

    static void lobatto_transpose_1d(int n_elem, int n_edges, int n_colloc, int n_var, FaceType face_type, const double * uf_, double * u_, const int * v2e_)
    {
        auto uf = reshape(uf_, n_var, 2, n_edges);
        auto u = reshape(u_, n_var, n_colloc * n_colloc * n_elem);
        auto v2e = reshape(v2e_, 2, n_edges);

        for (int e = 0; e < n_edges; ++e)
        {
            for (int side = 0; side < 2; ++side)
            {
                const int j = v2e(side, e);
                if (j >= 0)
                {
                    for (int d = 0; d < n_var; ++d)
                    {
                        u(d, j) += uf(d, side, e);
                    }
                }
                
            }
        }
    }

    void LobattoFaceProlongator::t(int n_var, const double * uf_, double * u_) const
    {
        if (n_edges == 0)
            return;

        switch (dim)
        {
        case 1:
            lobatto_transpose_1d(n_elem, n_edges, n_colloc, n_var, face_type, uf_, u_, _v2e);
            break;
        case 2:
            lobatto_transpose_2d(n_elem, n_edges, n_colloc, n_var, face_type, uf_, u_, _v2e);
            break;
        default:
            wdg_error("LobattoFaceProlongator::t not implemented for dim == " + std::to_string(dim));
            break;
        }
    }

// LegendreFaceProlongator
    LegendreFaceProlongator::LegendreFaceProlongator(const Mesh2D& mesh, const QuadratureRule * basis, FaceType edge_type_)
        : FaceProlongator(mesh, basis, edge_type_),
          P(n_colloc)
    {
        if (n_edges == 0)
            return;

        const double x = -1.0;
        lagrange_basis(P, n_colloc, basis->x, 1, &x);

        _v2e.reshape(2 * n_colloc * n_colloc * n_edges);
        _v2e.fill(-1);
        auto v2e = reshape(_v2e, n_colloc, n_colloc, 2, n_edges);

        const int nc = n_colloc;
        auto mapV2E = [nc](int k, int i, int f, int el) -> int
        {
            const int m = (f == 0 || f == 2) ? i : (f == 1) ? (nc-1-k) : k;
            const int n = (f == 1 || f == 3) ? i : (f == 2) ? (nc-1-k) : k;

            return m + nc * (n + nc * el);
        };

        for (int e = 0; e < n_edges; ++e)
        {
            const Edge * edge = mesh.edge(e, face_type);
            const int el0 = edge->elements[0];
            const int s0 = edge->sides[0];

            for (int i = 0; i < n_colloc; ++i)
            {
                for (int k = 0; k < n_colloc; ++k)
                {
                    v2e(k, i, 0, e) = mapV2E(k, i, s0, el0);
                }
            }

            if (face_type == FaceType::INTERIOR)
            {
                const int el1 = edge->elements[1];
                const int s1 = edge->sides[1];

                for (int i = 0; i < n_colloc; ++i)
                {
                    for (int k = 0; k < n_colloc; ++k)
                    {
                        const int j = (edge->delta > 0) ? i : (n_colloc-1-i);
                        v2e(k, i, 1, e) = mapV2E(k, j, s1, el1);
                    }
                }
            }
        }
    }

    LegendreFaceProlongator::LegendreFaceProlongator(const Mesh1D& mesh, const QuadratureRule * basis, FaceType edge_type_)
        : FaceProlongator(mesh, basis, edge_type_),
          P(n_colloc)
    {
        if (n_edges == 0)
            return;

        constexpr double x[] = {-1.0};
        lagrange_basis(P, n_colloc, basis->x, 1, x);

        _v2e.reshape(2 * n_colloc * n_edges);
        _v2e.fill(-1);
        auto v2e = reshape(_v2e, n_colloc, 2, n_edges);

        for (int e = 0; e < n_edges; ++e)
        {
            auto& face = mesh.face(e, face_type);

            const int el0 = face.elements[0];
            if (el0 >= 0)
            {
                for (int i = 0; i < n_colloc; ++i)
                {
                    v2e(i, 0, e) = (n_colloc-1-i) + n_colloc * el0;
                }
            }

            const int el1 = face.elements[1];
            if (el1 >= 0)
            {
                for (int i = 0; i < n_colloc; ++i)
                {
                    v2e(i, 1, e) = i + n_colloc * el1;
                }
            }
        }
    }

    static void legendre_action_2d(int n_elem, int n_edges, int n_colloc, int n_var, FaceType face_type, const double * u_, double * uf_, const int * v2e_, const dvec& P)
    {
        auto uf = reshape(uf_, n_colloc, n_var, 2, n_edges);
        auto u = reshape(u_, n_var, n_colloc * n_colloc * n_elem);
        auto v2e = reshape(v2e_, n_colloc, n_colloc, 2, n_edges);

        const int n_sides = (face_type == FaceType::INTERIOR) ? 2 : 1;

        uf.zeros();

        for (int e = 0; e < n_edges; ++e)
        {
            for (int side = 0; side < n_sides; ++side)
            {
                for (int i = 0; i < n_colloc; ++i)
                {
                    for (int d = 0; d < n_var; ++d)
                    {
                        double val = 0.0;
                        for (int k = 0; k < n_colloc; ++k)
                        {
                            const int j = v2e(k, i, side, e);
                            val += u(d, j) * P(k);
                        }
                        uf(i, d, side, e) = val;
                    }
                }
            }
        }
    }

    static void legendre_action_1d_interior(int n_elem, int n_edges, int n_colloc, int n_var, const double * u_, double * uf_, const int * v2e_, const dvec& P)
    {
        auto uf = reshape(uf_, n_var, 2, n_edges);
        auto u = reshape(u_, n_var, n_colloc * n_elem);
        auto v2e = reshape(v2e_, n_colloc, 2, n_edges);

        for (int e = 0; e < n_edges; ++e)
        {
            for (int side = 0; side < 2; ++side)
            {
                for (int d = 0; d < n_var; ++d)
                {
                    double val = 0.0;
                    for (int  k = 0; k < n_colloc; ++k)
                    {
                        const int j = v2e(k, side, e);
                        val += u(d, j) * P(k);
                    }
                    uf(d, side, e) = val;
                }
            }
        }
    }

    static void legendre_action_1d_boundary(int n_elem, int n_edges, int n_colloc, int n_var, const double * u_, double * uf_, const int * v2e_, const dvec& P)
    {
        auto uf = reshape(uf_, n_var, 2, n_edges);
        auto u = reshape(u_, n_var, n_colloc * n_elem);
        auto v2e = reshape(v2e_, n_colloc, 2, n_edges);

        for (int e = 0; e < n_edges; ++e)
        {
            for (int d = 0; d < n_var; ++d)
            {
                double val = 0.0;
                for (int k = 0; k < n_colloc; ++k)
                {
                    const int j = (v2e(k, 0, e) >= 0) ? v2e(k, 0, e) : v2e(k, 1, e);
                    val += u(d, j) * P(k);
                }
                uf(d, 0, e) = val;
                uf(d, 1, e) = val;
            }
        }
    }

    void LegendreFaceProlongator::action(int n_var, const double * u_, double * uf_) const
    {
        if (n_edges == 0)
            return;

        switch (dim)
        {
        case 1:
            if (face_type == FaceType::INTERIOR)
                legendre_action_1d_interior(n_elem, n_edges, n_colloc, n_var, u_, uf_, _v2e, P);
            else
                legendre_action_1d_boundary(n_elem, n_edges, n_colloc, n_var, u_, uf_, _v2e, P);
            break;
        case 2:
            legendre_action_2d(n_elem, n_edges, n_colloc, n_var, face_type, u_, uf_, _v2e, P);
            break;
        default:
            wdg_error("LegendreFaceProlongator::action not implemented for dim == " + std::to_string(dim));
            break;
        }
    }

    static void legendre_transpose_2d(int n_elem, int n_edges, int n_colloc, int n_var, FaceType face_type, const double * uf_, double * u_, const int * v2e_, const dvec& P)
    {
        auto uf = reshape(uf_, n_colloc, n_var, 2, n_edges);
        auto u = reshape(u_, n_var, n_colloc * n_colloc * n_elem);
        auto v2e = reshape(v2e_, n_colloc, n_colloc, 2, n_edges);

        const int n_sides = (face_type == FaceType::INTERIOR) ? 2 : 1;

        for (int e = 0; e < n_edges; ++e)
        {
            for (int side = 0; side < n_sides; ++side)
            {
                for (int i = 0; i < n_colloc; ++i)
                {
                    for (int d = 0; d < n_var; ++d)
                    {
                        double Y = uf(i, d, side, e);
                        for (int k = 0; k < n_colloc; ++k)
                        {
                            const int j = v2e(k, i, side, e);
                            u(d, j) += Y * P(k);
                        }
                    }
                }
            }
        }
    }

    static void legendre_transpose_1d(int n_elem, int n_edges, int n_colloc, int n_var, FaceType face_type, const double * uf_, double * u_, const int * v2e_, const dvec& P)
    {
        auto uf = reshape(uf_, n_var, 2, n_edges);
        auto u = reshape(u_, n_var, n_colloc * n_elem);
        auto v2e = reshape(v2e_, n_colloc, 2, n_edges);

        for (int e = 0; e < n_edges; ++e)
        {
            for (int side = 0; side < 2; ++side)
            {
                for (int d = 0; d < n_var; ++d)
                {
                    double Y = uf(d, side, e);
                    for (int k = 0; k < n_colloc; ++k)
                    {
                        const int j = v2e(k, side, e);
                        if (j >= 0)
                            u(d, j) += Y * P(k);
                    }
                }
            }
        }
    }

    void LegendreFaceProlongator::t(int n_var, const double * uf_, double * u_) const
    {
        if (n_edges == 0)
            return;
            
        switch (dim)
        {
        case 1:
            legendre_transpose_1d(n_elem, n_edges, n_colloc, n_var, face_type, uf_, u_, _v2e, P);
            break;
        case 2:
            legendre_transpose_2d(n_elem, n_edges, n_colloc, n_var, face_type, uf_, u_, _v2e, P);
            break;
        default:
            wdg_error("LobattoFaceProlongator::t not implemented for dim == " + std::to_string(dim));
            break;
        }
    }
#endif
} // namespace dg
