#include "WaveHoltz.hpp"

namespace dg
{
    WaveHoltz::WaveHoltz(double omega_, const Mesh2D& mesh, const QuadratureRule * basis, const int * boundary_conditions, bool approx_quad)
        : omega{omega_},
          ndof{3 * mesh.n_elem() * basis->n * basis->n},
          rk(ndof),
          p(ndof)
    {
        if (approx_quad)
        {
            m.reset(new MassMatrix<true>(3, mesh, basis));
            a.reset(new WaveEquation<true>(mesh, basis));
            bc.reset(new WaveBC<true>(mesh, boundary_conditions, basis));
        }
        else
        {
            m.reset(new MassMatrix<false>(3, mesh, basis));
            a.reset(new WaveEquation<false>(mesh, basis));
            bc.reset(new WaveBC<false>(mesh, boundary_conditions, basis));
        }

        T = 2.0 * M_PI / omega;
        double h = mesh.min_edge_measure();
        double p = basis->n;
        dt = 0.5 * h / (p * p);
        nt = std::ceil(T / dt);
        dt = T / nt;
    }

    void WaveHoltz::pi0(double * u, const double * F) const
    {
        auto time_derivative = [this, F](double * vt, const double t, const double * v) -> void
        {
            for (int i=0; i < ndof; ++i)
                vt[i] = 0.0;
                
            a->action(v, vt);
            bc->action(v, vt);

            const double s = -1.0 / omega * std::sin(omega * t);
            for (int i=0; i < ndof; ++i)
                vt[i] += F[i] * s;

            m->inv(vt);
        };

        for (int i=0; i < ndof; ++i)
        {
            p(i) = 0.0;
            u[i] = 0.0;
        }
        
        double t = 0.0;
        for (int it=1; it <= nt; ++it)
        {
            rk.step(dt, time_derivative, t, p);

            const double c = ((it == nt) ? 0.5 : 1.0) * dt * K(t);
            for (int i=0; i < ndof; ++i)
            {
                u[i] += c * p(i);
            }
        }
    }

    void WaveHoltz::S(double * u) const
    {
        auto time_derivative = [this](double * vt, const double t, const double * v) -> void
        {
            for (int i=0; i < ndof; ++i)
                vt[i] = 0.0;
            
            a->action(v, vt);
            bc->action(v, vt);

            m->inv(vt);
        };

        for (int i=0; i < ndof; ++i)
        {
            p(i) = u[i];
            u[i] *= 0.5 * dt * K(0.0);
        }

        double t = 0.0;
        for (int it=1; it <= nt; ++it)
        {
            rk.step(dt, time_derivative, t, p);

            const double c = ((it == nt) ? 0.5 : 1.0) * dt * K(t);
            for (int i=0; i < ndof; ++i)
            {
                u[i] += c * p(i);
            }
        }
    }

    void WaveHoltz::postprocess(double * H_, const double * U_) const
    {
        const int n = ndof / 3;

        // H = p + i * (1/omega * div(u))
        auto H = reshape(H_, 2, n);
        auto U = reshape(U_, 3, n);

        // imag(H) = 1/omega * div(u)
        p.zeros();
        a->action(U, p);
        bc->action(U, p);
        m->inv(p);

        auto P = reshape(p, 3, n);

        for (int i=0; i < n; ++i)
        {
            H(0, i) = U(0, i);
            H(1, i) = 1.0 / omega * P(0, i);
        }
    }
} // namespace dg
