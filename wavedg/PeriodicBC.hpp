#ifndef DG_PERIODIC_BC_HPP
#define DG_PERIODIC_BC_HPP

#include "wdg_config.hpp"
#include "FaceProlongator.hpp"
#include "Mesh1D.hpp"
#include "Operator.hpp"
#include "EdgeFlux.hpp"

namespace dg
{
    class PeriodicBC1d : public Operator
    {
    public:
        PeriodicBC1d(int n_var, const Mesh1D& mesh, const QuadratureRule * basis, const double * a, bool constant_coefficient);

        ~PeriodicBC1d() = default;

        void action(const double * u, double * divF) const override;

    private:
        const int n_var;

        std::unique_ptr<FaceProlongator> face_prol;
        std::unique_ptr<Operator> flx;

        mutable dvec uB;
    };
} // namespace dg


#endif