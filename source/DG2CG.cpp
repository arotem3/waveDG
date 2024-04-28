#include "DG2CG.hpp"

static void setup_mass(int n_elem, int n_basis, const double * w_, const double * detJ_, double * m_)
{
    auto w = dg::reshape(w_, n_basis);
    auto detJ = dg::reshape(detJ_, n_basis, n_basis, n_elem);
    auto m = dg::reshape(m_, n_basis, n_basis, n_elem);

    for (int el = 0; el < n_elem; ++el)
        for (int j = 0; j < n_basis; ++j)
            for (int i = 0; i < n_basis; ++i)
                m(i,j,el) = w(i) * w(j) * detJ(i,j,el);
}

namespace dg
{
    void DG2CG::action(int n_var, double * u_) const
    {
        auto u = reshape(u_, n_var, n_dof);
        
        for (int i = 0; i < n_dof; ++i)
        {
            const double mi = m(i);
            for (int d = 0; d < n_var; ++d)
                u(i, d) *= mi;
        }

        cgm.sum(n_var, u);
        cgm.unmask(n_var, u);

        for (int i = 0; i < n_dof; ++i)
        {
            const double mi = inv_m(i);
            for (int d = 0; d < n_var; ++d)
                u(d, i) *= mi;
        }
    }

    DG2CG::DG2CG(const CGMask& cgm_, const Mesh2D& mesh, const QuadratureRule * basis)
        : n_dof{mesh.n_elem() * basis->n * basis->n},
          cgm{cgm_},
          m(n_dof),
          inv_m(n_dof)
    {
        const double * w = basis->w;

        auto& metrics = mesh.element_metrics(basis);
        const double * detJ = metrics.measures();

        setup_mass(mesh.n_elem(), basis->n, w, detJ, m);

        copy(n_dof, m, inv_m);
        cgm.sum(1, inv_m);
        cgm.unmask(1, inv_m);
        for (int i = 0; i < n_dof; ++i)
            inv_m(i) = 1.0 / inv_m(i);
    }
} // namespace dg
