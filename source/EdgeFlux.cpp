#include "EdgeFlux.hpp"

static bool approx_symmetric(int n, const double * a)
{
    constexpr double eps = std::numeric_limits<double>::epsilon();
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < i; ++i)
        {
            double e = std::abs(a[i + n*j] - a[j + n*i]);
            double rtol = eps * std::max(std::abs(a[i + n*j]), 1.0);
            if (e < rtol)
                return false;
        }
    }
    return true;
}

namespace dg
{
    class _ComputeFlux
    {
    private:
        const int m;
        const double a;
        const double b;

        mutable dmat R;
        mutable dvec e;

    public:
        _ComputeFlux(int nvar, double a_, double b_) : m(nvar), a(a_), b(b_), R(m, m), e(m) {}

        void flux(const double * nA, double * F_, int side) const
        {
            const double sgn = (side == 0) ? 1.0 : -1.0;

            if (m == 1)
            {
                *F_ = 0.5*a*(*nA) + sgn*b*std::abs(*nA);
                return;
            }

            if (not approx_symmetric(m, nA))
                throw std::runtime_error("EdgeFlux only supports symmetric n.A");

            eig(m, R.data(), e.data(), nA);
                
            auto F = reshape(F_, m, m);

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

            // F <- R F
            for (int j = 0; j < m; ++j)
            {
                // e <- R * F(:, j)
                for (int i = 0; i < m; ++i)
                {
                    e[i] = 0.0;
                    for (int k = 0; k < m; ++k)
                    {
                        e[i] += R(i, k) * F(k, j);
                    }
                }
                
                // F(:, j) <- e
                for (int i = 0; i < m; ++i)
                {
                    F(i, j) = e[i];
                }
            }
        }    
    };

    EdgeFlux::EdgeFlux(int nvar, const Mesh2D& mesh, EdgeType edge_type, const QuadratureRule * basis, const double * nA_, double a, double b)
        : etype(edge_type),
          n_edges(mesh.n_edges(edge_type)),
          n_quad(basis->n),
          n_colloc(basis->n),
          n_var(nvar),
          use_colloc(true),
          P(),
          F(n_var, n_var, 2, n_quad, n_edges),
          quad(basis),
          uf(n_var, 2),
          Uq(n_var, n_colloc, 2)
    {
        _ds = mesh.edge_measures(basis, etype);

        auto nA = reshape(nA_, n_var*n_var, n_colloc, 2, n_edges);

        _ComputeFlux flx(n_var, a, b);
        for (int e = 0; e < n_edges; ++e)
        {
            for (int i = 0; i < n_colloc; ++i)
            {
                for (int s = 0; s < 2; ++s)
                {
                    const double * nAs = &nA(0, i, s, e);
                    double * Fs = &F(0, 0, s, i, e);

                    flx.flux(nAs, Fs, s);
                }
            }
        }
    }

    EdgeFlux::EdgeFlux(int nvar, const Mesh2D& mesh, EdgeType edge_type, const QuadratureRule * basis, const QuadratureRule * quad_, const double * nA_, double a, double b)
        : etype(edge_type),
          n_edges(mesh.n_edges(edge_type)),
          n_quad(quad_->n),
          n_colloc(basis->n),
          n_var(nvar),
          use_colloc(false),
          P(n_quad, n_colloc),
          F(n_var, n_var, 2, n_quad, n_edges),
          quad(quad_),
          uf(n_var, 2),
          Uq(n_var, n_quad, 2)
    {
        lagrange_basis(P.data(), n_colloc, basis->x, n_quad, quad->x);

        _ds = mesh.edge_measures(quad, etype);

        dvec nAq(n_var*n_var);
        auto nA = reshape(nA_, n_var*n_var, n_colloc, 2, n_edges);

        _ComputeFlux flx(n_var, a, b);
        for (int e = 0; e < n_edges; ++e)
        {
            for (int i = 0; i < n_quad; ++i)
            {
                // interpolate n.A to quadrature point
                for (int s = 0; s < 2; ++s)
                {
                    for (int d = 0; d < n_var*n_var; ++d)
                    {
                        double Aq = 0.0;
                        for (int j = 0; j < n_colloc; ++j)
                        {
                            Aq += nA(d, j, s, e) * P(i, j);
                        }
                        nAq(d) = Aq;
                    }

                    double * Fs = &F(0, 0, s, i, e);
                    flx.flux(nAq.data(), Fs, s);
                }
            }
        }
    }

    void EdgeFlux::operator()(double * ub_) const
    {
        auto ub = reshape(ub_, n_var, n_colloc, 2, n_edges);
        auto ds = reshape(_ds, n_quad, n_edges);

        if (use_colloc)
        {
            for (int e = 0; e < n_edges; ++e)
            {
                for (int i = 0; i < n_colloc; ++i)
                {
                    const double w = quad->w[i] * ds(i, e);
                    
                    // copy u|face to work arrays
                    for (int d = 0; d < n_var; ++d)
                    {
                        uf(d, 0) = ub(d, i, 0, e);
                        uf(d, 1) = ub(d, i, 1, e);
                    }

                    // evaluate flux and integrate
                    for (int d = 0; d < n_var; ++d)
                    {
                        double flx = 0.0;

                        for (int k = 0; k < n_var; ++k)
                        {
                            flx += F(d, k, 0, i, e) * uf(k, 0) + F(d, k, 1, i, e) * uf(k, 1);
                        }
                        flx *= w;

                        ub(d, i, 0, e) =  flx;
                        ub(d, i, 1, e) = -flx;
                    }
                }
            }
        }
        else
        {
            for (int e = 0; e < n_edges; ++e)
            {
                // evaluate u at quadrature points
                for (int s = 0; s < 2; ++s)
                {
                    for (int i = 0; i < n_quad; ++i)
                    {
                        for (int d = 0; d < n_var; ++d)
                        {
                            double u = 0.0;
                            for (int j = 0; j < n_colloc; ++j)
                            {
                                u += ub(d, j, s, e) * P(i, j);
                            }
                            Uq(d, i, s) = u;
                        }
                    }
                }

                // evaluate fluxes
                for (int i = 0; i < n_quad; ++i)
                {
                    // copy u|face to work arrays
                    for (int d = 0; d < n_var; ++d)
                    {
                        uf(d, 0) = Uq(d, i, 0);
                        uf(d, 1) = Uq(d, i, 1);
                    }

                    // eval flux
                    for (int d = 0; d < n_var; ++d)
                    {
                        double flx = 0.0;
                        for (int k = 0; k < n_var; ++k)
                        {
                            flx += F(d, k, 0, i, e) * uf(k, 0) + F(d, k, 1, i, e) * uf(k, 1);
                        }
                        Uq(d, i, 0) = flx;
                    }
                }

                // integrate
                for (int i = 0; i < n_colloc; ++i)
                {
                    for (int d = 0; d < n_var; ++d)
                    {
                        double flx = 0.0;
                        for (int j = 0; j < n_quad; ++j)
                        {
                            flx += Uq(d, j, 0) * P(j, i) * quad->w[j] * ds(j, e);
                        }
                        ub(d, i, 0, e) =  flx;
                        ub(d, i, 1, e) = -flx;
                    }
                }
            }
        }
    }
} // namespace dg
