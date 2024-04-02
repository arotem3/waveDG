#include "WaveEquation.hpp"

namespace dg
{
    WaveEquation::WaveEquation(const Mesh2D& mesh, const QuadratureRule * basis, bool approx_quad, const QuadratureRule * quad)
        : dim(2), uI(3, mesh, FaceType::INTERIOR, basis)
    {
        prol = make_face_prolongator(mesh, basis, FaceType::INTERIOR);

        const double a[] = {
            0.0, 1.0, 0.0,
            1.0, 0.0, 0.0,
            0.0, 0.0, 0.0,

            0.0, 0.0, 1.0,
            0.0, 0.0, 0.0,
            1.0, 0.0, 0.0
        }; // wave equation as conservation law: p_t + div(u) == 0, u_t + grad(u) == 0.

        if (approx_quad)
        {
            div.reset(new Div<true>(3, mesh, basis, a, true));
            flx.reset(new EdgeFlux<true>(3, mesh, FaceType::INTERIOR, basis, a, true));
        }
        else
        {
            div.reset(new Div<false>(3, mesh, basis, a, true, quad));
            flx.reset(new EdgeFlux<false>(3, mesh, FaceType::INTERIOR, basis, a, true, -1.0, -0.5, quad));
        }
    }

    WaveEquation::WaveEquation(const Mesh1D& mesh, const QuadratureRule * basis, bool approx_quad, const QuadratureRule * quad)
        : dim(1), uI(2, mesh, FaceType::INTERIOR, basis)
    {
        prol = make_face_prolongator(mesh, basis, FaceType::INTERIOR);

        const double a[] = {
            0.0, 1.0,
            1.0, 0.0,
        }; // wave equation as conservation law: p_t + div(u) == 0, u_t + grad(u) == 0.

        if (approx_quad)
        {
            div.reset(new Div<true>(2, mesh, basis, a, true));
            flx.reset(new EdgeFlux<true>(2, mesh, FaceType::INTERIOR, basis, a, true));
        }
        else
        {
            div.reset(new Div<false>(2, mesh, basis, a, true, quad));
            flx.reset(new EdgeFlux<false>(2, mesh, FaceType::INTERIOR, basis, a, true, -1.0, -0.5, quad));
        }
    }

    void WaveEquation::action(const double * u, double * divF) const
    {
        const int n_var = 1 + dim;

        div->action(u, divF);

        prol->action(n_var, u, uI);

        #ifdef WDG_USE_MPI
        uI.send_recv();
        #endif

        flx->action(uI, uI);
        prol->t(n_var, uI, divF);
    }

    WaveBC::WaveBC(const Mesh2D& mesh, const int * bc_, const QuadratureRule * basis, bool approx_quad, const QuadratureRule * quad)
        : dim(2),
          nB(mesh.n_edges(FaceType::BOUNDARY)),
          n_colloc(basis->n),
          bc(nB),
          uB(3, mesh, FaceType::BOUNDARY, basis)
    {
        for (int i=0; i < nB; ++i)
        {
        #ifdef WDG_DEBUG
            if ((bc_[i] != 0) && (bc_[i] != 1))
                wdg_error("WaveBC error: specified boundary condition not implemented.");
        #endif
            bc(i) = bc_[i];
        }

        const double * n_ = mesh.edge_metrics(basis, FaceType::BOUNDARY).normals();
        normals_2d = reshape(n_, 2, n_colloc, nB);

        prol = make_face_prolongator(mesh, basis, FaceType::BOUNDARY);

        const double a[] = {
            0.0, 1.0, 0.0,
            1.0, 0.0, 0.0,
            0.0, 0.0, 0.0,

            0.0, 0.0, 1.0,
            0.0, 0.0, 0.0,
            1.0, 0.0, 0.0
        };

        if (approx_quad)
            flx.reset(new EdgeFlux<true>(3, mesh, FaceType::BOUNDARY, basis, a, true));
        else
            flx.reset(new EdgeFlux<false>(3, mesh, FaceType::BOUNDARY, basis, a, true, -1.0, -0.5, quad));
    }

    WaveBC::WaveBC(const Mesh1D& mesh, const int * bc_, const QuadratureRule * basis)
        : dim(1),
          nB(mesh.n_faces(FaceType::BOUNDARY)),
          n_colloc(basis->n),
          normals_1d(nB),
          bc(nB),
          uB(2, mesh, FaceType::BOUNDARY, basis)
    {
        for (int i = 0; i < nB; ++i)
        {
        #ifdef WDG_DEBUG
            if ((bc_[i] != 0) && (bc_[i] != 1))
                wdg_error("WaveBC error: specified boundary condition not implemented.");
        #endif
            bc(i) = bc_[i];
        }

        for (int f = 0; f < nB; ++f)
        {
            auto& face = mesh.face(f, FaceType::BOUNDARY);

            if (face.elements[0] < 0) // left boundary
                normals_1d(f) = -1.0;
            else // right boundary
                normals_1d(f) = 1.0;
        }

        prol = make_face_prolongator(mesh, basis, FaceType::BOUNDARY);

        const double a[] = {
            0.0, 1.0,
            1.0, 0.0,
        };

        flx.reset(new EdgeFlux<true>(2, mesh, FaceType::BOUNDARY, basis, a, true));
    }

    static void apply_bc_2d(int nB, int n_colloc, double * uB_, const ivec& bc, const const_dcube_wrapper& n)
    {
        auto uB = reshape(uB_, n_colloc, 3, 2, nB);

        // compute exterior values
        double v_ext[3], v_int[3];
        for (int e=0; e < nB; ++e)
        {
            auto compute_exterior_values = (bc(e) == 0) ? absorb_2d : reflect_2d;
            
            for (int i=0; i < n_colloc; ++i)
            {
                v_int[0] = uB(i, 0, 0, e);
                v_int[1] = uB(i, 1, 0, e);
                v_int[2] = uB(i, 2, 0, e);

                compute_exterior_values(v_ext, v_int, &n(0, i, e));

                uB(i, 0, 1, e) = v_ext[0];
                uB(i, 1, 1, e) = v_ext[1];
                uB(i, 2, 1, e) = v_ext[2];
            }
        }
    }

    static void apply_bc_1d(int nB, double * uB_, const ivec& bc, const dvec& n)
    {
        auto uB = reshape(uB_, 2, 2, nB);

        // compute exterior values
        double v_ext[2], v_int[2];
        for (int e = 0; e < nB; ++e)
        {
            auto compute_exterior_values = (bc(e) == 0) ? absorb_1d : reflect_1d;

            const int s = (n(e) < 0) ? 1 : 0;
            v_int[0] = uB(0, s, e);
            v_int[1] = uB(1, s, e);

            compute_exterior_values(v_ext, v_int, n(e));

            uB(0, 1-s, e) = v_ext[0];
            uB(1, 1-s, e) = v_ext[1];
        }
    }

    void WaveBC::action(const double * u, double * divF) const
    {
        const int n_var = dim + 1;

        // prolongate face values
        prol->action(n_var, u, uB);

        switch (dim)
        {
        case 1:
            apply_bc_1d(nB, uB, bc, normals_1d);
            break;
        case 2:
            apply_bc_2d(nB, n_colloc, uB, bc, normals_2d);
            break;
        default:
            wdg_error("");
            break;
        }

        // flux
        flx->action(uB, uB);

        // add to divF
        prol->t(n_var, uB, divF);
    }
} // namespace dg
