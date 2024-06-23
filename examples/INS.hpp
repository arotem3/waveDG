#ifndef WDG_INS_HPP
#define WDG_INS_HPP

#include "wavedg.hpp"

using namespace dg;

class PressureElliptic : public Operator
{
public:
    PressureElliptic(int p_dof, const Operator * stiffness, const CGMask& masker, const ZeroBoundary& DirichletBC)
        : S(stiffness), cgm(masker), Zp(DirichletBC), z(p_dof) {}

    void action(const double * p, double * Ap) const override
    {
        copy(z.size(), p, z);
        Zp.action(z);
        cgm.unmask(1, z);

        zeros(z.size(), Ap);
        S->action(z, Ap);

        cgm.sum(1, Ap);
        cgm.mask(1, Ap);
        Zp.action(Ap);
    }

    void action(int n_var, const double * p, double * Ap) const override
    {
        wdg_error("PressureElliptic does not define the action function with variable n_var.");
    }

    void residual(const double * p, double * b) const
    {
        const int n = z.size();
        dvec Ap(n);
        action(p, Ap);

        cgm.sum(1, b);
        Zp.action(1, b);

        for (int i=0; i < n; ++i)
            b[i] -= Ap[i];

        cgm.mask(1, b);
    }

    void postprocess(double * p) const
    {
        cgm.unmask(1, p);
    }

private:
    const Operator * S;
    const CGMask& cgm;
    const ZeroBoundary& Zp;
    mutable dvec z;
};

class VelocityElliptic : public Operator
{
public:
    VelocityElliptic(int u_dof, const Operator * mass, const Operator * stiffness, const CGMask& masker, const ZeroBoundary& DiricheletBC)
        : M(mass), S(stiffness), cgm(masker), Zu(DiricheletBC), z(u_dof) {}

    void set_alpha(double a)
    {
        alpha = a;
    }

    void action(const double * u, double * Au) const override
    {
        const int n = z.size();
        copy(n, u, z);
        Zu.action(2, z);
        cgm.unmask(2, z);

        zeros(n, Au);
        M->action(2, z, Au);
        for (int i=0; i < n; ++i)
            Au[i] *= alpha;
        S->action(2, z, Au);

        cgm.sum(2, Au);
        cgm.mask(2, Au);
        Zu.action(2, Au);
    }

    void action(int n_var, const double * u, double * Au) const override
    {
        wdg_error("VelocityElliptic does not define action with variable n_var");
    }

    void prepare(double c, const double * G, double * u, double * b) const
    {
        const int n = z.size();
        dvec Ag(n);
        action(G, Ag);

        cgm.sum(2, b);
        cgm.mask(2, b);
        Zu.action(2, b);

        for (int i=0; i < n; ++i)
            b[i] = alpha * b[i] - c * Ag[i];

        Zu.action(2, u);
        cgm.mask(2, u);

        action(u, Ag);
        for (int i=0; i < n; ++i)
            b[i] -= Ag[i];
    }

    void post_process(double c, const double * G, double * u) const
    {
        const int n = z.size();
        for (int i=0; i < n; ++i)
            u[i] += c * G[i];
        cgm.unmask(2, u);
    }

private:
    double alpha;
    const Operator * M;
    const Operator * S;
    const CGMask& cgm;
    const ZeroBoundary& Zu;
    mutable dvec z;
};

#endif