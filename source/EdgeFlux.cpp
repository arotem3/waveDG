#include "EdgeFlux.hpp"

extern "C" void dgels_(char * TRANS, const int * M, const int * N, const int * NRHS, double * A, const int * LDA, double * B, const int * LDB, double * WORK, int * LWORK, int * INFO);

namespace dg
{
    namespace
    {
        class _ComputeFlux
        {
        private:
            const int m;
            const double a;
            const double b;

            mutable dmat R;
            mutable dvec e;
            mutable std::vector<double> work;

        public:
            _ComputeFlux(int nvar, double a_, double b_) : m(nvar), a(a_), b(b_), R(m, m), e(m), work(1)
            {
                char trans = 'T';
                int lwork = -1;
                int info;

                dgels_(&trans, &m, &m, &m, nullptr, &m, nullptr, &m, work.data(), &lwork, &info);

                lwork = work[0];
                work.resize(lwork);
            }

            void flux(const double * nA, double * F_, int side) const
            {
                const double sgn = (side == 0) ? 1.0 : -1.0;

                if (m == 1)
                {
                    *F_ = 0.5*a*(*nA) + sgn*b*std::abs(*nA);
                    return;
                }

                auto F = reshape(F_, m, m);

                if (b == 0) // F <- a/2 nA'
                {
                    auto n_A = reshape(nA, m, m);
                    for (int j=0; j < m; ++j)
                    {
                        for (int i=0; i < m; ++i)
                        {
                            F(i, j) = 0.5 * a * n_A(j, i);
                        }
                    }

                    return;
                }

                bool success = real_eig(m, R.data(), e.data(), nA);
                if (not success)
                    wdg_error("EdgeFlux error: failed to computed real eigenvalue decomposition of matrix n.A in computation of numerical flux.");

                // F <- R'
                for (int i=0; i < m; ++i)
                {
                    for (int j = 0; j < m; ++j)
                    {
                        F(i, j) = R(j, i);
                    }
                }

                // F <- diag(a/2 e + sgn b |e|) F
                for (int i=0; i < m; ++i)
                {
                    const double c = 0.5*a * e(i) + sgn * b * std::abs(e(i));
                    for (int j = 0; j < m; ++j)
                    {
                        F(i, j) *= c;
                    }
                }

                // F <- R^{-T} F
                char trans = 'T';
                int lwork = work.size();
                int info;

                dgels_(&trans, &m, &m, &m, R.data(), &m, F.data(), &m, work.data(), &lwork, &info);
                if (info != 0)
                    wdg_error("EdgeFlux error: eigenvectors of n.A linearly dependant, cannot compute flux.");
            }
        };
    } // namespace

    template <>
    EdgeFlux<true>::EdgeFlux(int nvar, const Mesh2D& mesh, FaceType edge_type, const QuadratureRule * basis, const double * A_, bool constant_coefficient, double a, double b, const QuadratureRule * quad_)
        : etype(edge_type),
          n_edges(mesh.n_edges(edge_type)),
          n_colloc(basis->n),
          n_var(nvar),
          F(n_colloc, n_var, n_var, 2, n_edges),
          uf(n_colloc, n_var)
    {
        auto& metrics = mesh.edge_metrics(basis, etype);
        const double * _ds = metrics.measures();
        auto ds = reshape(_ds, n_colloc, n_edges);

        const double * _n = metrics.normals();
        auto n = reshape(_n, 2, n_colloc, n_edges);

        auto W = reshape(basis->w, n_colloc);

        const int v2d = n_var * n_var;
        dvec nA(v2d);
        dmat Fs(n_var, n_var);

        _ComputeFlux flx(n_var, a, b);
        if (constant_coefficient)
        {
            auto A = reshape(A_, v2d, 2);

            for (int e = 0; e < n_edges; ++e)
            {
                for (int s = 0; s < 2; ++s)
                {
                    for (int i = 0; i < n_colloc; ++i)
                    {
                        for (int d = 0; d < v2d; ++d)
                        {
                            nA(d) = n(0, i, e) * A(d, 0) + n(1, i, e) * A(d, 1);
                        }

                        flx.flux(nA, Fs, s);

                        double w = W(i) * ds(i, e);
                        for (int d = 0; d < n_var; ++d)
                        {
                            for (int c = 0; c < n_var; ++c)
                            {
                                F(i, c, d, s, e) = w * Fs(c, d);
                            }
                        }
                    }
                }
            }
        }
        else
        {
            auto A = reshape(A_, n_colloc, v2d, 2, 2, n_edges);

            for (int e = 0; e < n_edges; ++e)
            {
                for (int s = 0; s < 2; ++s)
                {
                    for (int i = 0; i < n_colloc; ++i)
                    {
                        for (int d = 0; d < v2d; ++d)
                        {
                            nA(d) = n(0, i, e) * A(i, d, 0, s, e) + n(1, i, e) * A(i, d, 1, s, e);
                        }

                        flx.flux(nA, Fs, s);

                        double w = W(i) * ds(i, e);
                        for (int d = 0; d < n_var; ++d)
                        {
                            for (int c = 0; c < n_var; ++c)
                            {
                                F(i, c, d, s, e) = w * Fs(c, d);
                            }
                        }
                    }
                }
            }
        }
    }

    template <>
    void EdgeFlux<true>::action(const double * ub_, double * fb_) const
    {
        auto ub = reshape(ub_, n_colloc, n_var, 2, n_edges);
        auto fb = reshape(fb_, n_colloc, n_var, 2, n_edges);

        for (int e = 0; e < n_edges; ++e)
        {
            uf.zeros();

            for (int s = 0; s < 2; ++s)
            {
                // evaluate flux
                for (int d = 0; d < n_var; ++d)
                {
                    for (int c = 0; c < n_var; ++c)
                    {
                        for (int i = 0; i < n_colloc; ++i)
                        {
                            uf(i, d) += F(i, c, d, s, e) * ub(i, c, s, e);
                        }
                    }
                }
            }

            for (int d = 0; d < n_var; ++d)
            {
                for (int i = 0; i < n_colloc; ++i)
                {
                    fb(i, d, 0, e) =  uf(i, d);
                    fb(i, d, 1, e) = -uf(i, d);
                }
            }
        }
    }

    template <>
    EdgeFlux<false>::EdgeFlux(int nvar, const Mesh2D& mesh, FaceType edge_type, const QuadratureRule * basis, const double * A_, bool constant_coefficient, double a, double b, const QuadratureRule * quad)
        : etype(edge_type),
          n_edges(mesh.n_edges(edge_type)),
          n_colloc(basis->n),
          n_var(nvar)
    {
        if (quad == nullptr)
        {
            int p = 2*(n_colloc-1) + 2*mesh.max_element_order();
            if (not constant_coefficient)
            {
                p += n_colloc-1;
            }
            p = 1 + p/2;

            quad = QuadratureRule::quadrature_rule(p);
        }
        n_quad = quad->n;

        P.reshape(n_quad, n_colloc);
        Pt.reshape(n_colloc, n_quad);
        F.reshape(n_var, n_var, n_quad, 2, n_edges);
        Uq.reshape(n_var, n_quad);
        uf.reshape(n_quad, n_var);

        lagrange_basis(P, n_colloc, basis->x, n_quad, quad->x);
        for (int j=0; j < n_colloc; ++j)
        {
            for (int i=0; i < n_quad; ++i)
            {
                Pt(j, i) = P(i, j);
            }
        }

        auto& metrics = mesh.edge_metrics(quad, etype);
        const double * _ds = metrics.measures();
        auto ds = reshape(_ds, n_quad, n_edges);

        const double * _n = metrics.normals();
        auto n = reshape(_n, 2, n_quad, n_edges);

        auto W = reshape(quad->w, n_quad);

        const int v2d = n_var * n_var;
        dvec nA(v2d);
        dmat Fs(n_var, n_var);

        _ComputeFlux flx(n_var, a, b);
        if (constant_coefficient)
        {
            auto A = reshape(A_, v2d, 2);

            for (int e = 0; e < n_edges; ++e)
            {
                for (int i = 0; i < n_quad; ++i)
                {
                    for (int s = 0; s < 2; ++s)
                    {
                        for (int d = 0; d < v2d; ++d)
                        {
                            nA(d) = n(0, i, e) * A(d, 0) + n(1, i, e) * A(d, 1);
                        }

                        flx.flux(nA, Fs, s);

                        double w = W(i) * ds(i, e);
                        for (int d = 0; d < n_var; ++d)
                        {
                            for (int c = 0; c < n_var; ++c)
                            {
                                F(c, d, i, s, e) = w * Fs(c, d);
                            }
                        }
                    }
                }
            }
        }
        else
        {
            auto A = reshape(A_, 2, v2d, 2, n_colloc, n_edges);

            for (int e = 0; e < n_edges; ++e)
            {
                for (int i = 0; i < n_quad; ++i)
                {
                    for (int s = 0; s < 2; ++s)
                    {
                        for (int d = 0; d < v2d; ++d)
                        {
                            double a0 = 0.0;
                            double a1 = 0.0;
                            for (int j = 0; j < n_colloc; ++j)
                            {
                                double p = Pt(j, i);
                                a0 += A(s, d, 0, j, e) * p;
                                a1 += A(s, d, 1, j, e) * p;
                            }
                            nA(d) = n(0, i, e) * a0 + n(1, i, e) * a1;
                        }

                        flx.flux(nA, Fs, s);

                        double w = W(i) * ds(i, e);
                        for (int d = 0; d < n_var; ++d)
                        {
                            for (int c = 0; c < n_var; ++c)
                            {
                                F(c, d, i, s, e) = w * Fs(c, d);
                            }
                        }
                    }
                }
            }
        }
    }

    template <>
    void EdgeFlux<false>::action(const double * ub_, double * fb_) const
    {
        auto ub = reshape(ub_, n_colloc, n_var, 2, n_edges);
        auto fb = reshape(fb_, n_colloc, n_var, 2, n_edges);

        for (int e = 0; e < n_edges; ++e)
        {
            uf.zeros();

            for (int s = 0; s < 2; ++s)
            {
                // evaluate u at quadrature points
                for (int i = 0; i < n_quad; ++i)
                {
                    for (int d = 0; d < n_var; ++d)
                    {
                        double u = 0.0;
                        for (int j = 0; j < n_colloc; ++j)
                        {
                            u += ub(j, d, s, e) * Pt(j, i);
                        }
                        Uq(d, i) = u;
                    }
                }

                // evaluate fluxes
                for (int i = 0; i < n_quad; ++i)
                {
                    for (int d = 0; d < n_var; ++d)
                    {
                        double flx = 0.0;
                        for (int c = 0; c < n_var; ++c)
                        {
                            flx += F(c, d, i, s, e) * Uq(c, i);
                        }
                        uf(i, d) += flx;
                    }
                }
            }

            // integrate
            for (int d = 0; d < n_var; ++d)
            {
                for (int i = 0; i < n_colloc; ++i)
                {
                    double flx = 0.0;
                    for (int j = 0; j < n_quad; ++j)
                    {
                        flx += uf(j, d) * P(j, i);
                    }
                    fb(i, d, 0, e) =  flx;
                    fb(i, d, 1, e) = -flx;
                }
            }
        }
    }

} // namespace dg
