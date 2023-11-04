#include "QuadratureRule.hpp"

// extern to lapack routine dsteqr for eigevalue decomposition of symmetric
// tridiagonal matrix:
// COMPZ: is a single character, set to 'N' for only eigenvalues
// N: order of matrix
// D: pointer to matrix diagonal (length n)
// E: pointer to matrix off diagonal (length n-1)
// Z: pointer to orthogonal matrix, not necessary here
// LDZ_dummy: leading dim of Z, just set to 1 since it wont be used
// WORK: pointer to array, not needed for COMPZ=='N'
// INFO: 0 on success, <0 failed, >0 couldn't find all eigs
extern "C" int dsteqr_(char* COMPZ, int* N, double* D, double* E, double* Z, int* LDZ_dummy, double* WORK, int* INFO);

static inline double square(double x)
{
    return x * x;
}

namespace dg
{
    QuadratureRule::QuadratureRule(int n_, QuadratureType type_) : _x(new double[n_]), _w(new double[n_]), n(n_), x(_x.get()), w(_w.get()), type(type_) {}

    QuadratureRule::QuadratureRule(const QuadratureRule& q) : _x(new double[q.n]), _w(new double[q.n]), n(q.n), x(_x.get()), w(_w.get()), type(q.type)
    {
        std::copy_n(q.x, n, _x.get());
        std::copy_n(q.w, n, _w.get());
    }

    QuadratureRule::QuadratureRule(QuadratureRule&& q) : _x(std::move(q._x)), _w(std::move(q._w)), n(q.n), x(_x.get()), w(_w.get()), type(q.type) {}

    QuadratureRule QuadratureRule::gauss_lobatto(int n)
    {
        if (n < 2)
            throw std::invalid_argument("gauss_lobatto error: require n >= 2, but n =" + std::to_string(n) + ".");
            
        static const std::unordered_map<int, std::vector<double>> cached_nodes = {
            {2, {-1,1}},
            {3, {-1,0,1}},
            {4, {-1, -0.447213595499958, 0.447213595499958, 1}},
            {5, {-1, -0.654653670707977, 0, 0.654653670707977, 1}},
            {6, {-1, -0.765055323929465, -0.285231516480645, 0.285231516480645, 0.765055323929465, 1}},
            {7, {-1, -0.830223896278567, -0.468848793470714, 0.0, 0.468848793470714, 0.830223896278567, 1}},
            {8, {-1, -0.871740148509607, -0.591700181433142, -0.209299217902479, 0.2092992179024789, 0.591700181433142, 0.871740148509607, 1}},
            {9, {-1, -0.899757995411460, -0.677186279510738, -0.363117463826178, 0, 0.363117463826178, 0.677186279510738, 0.899757995411460, 1}}
        };
        
        QuadratureRule q(n, QuadratureRule::GaussLobatto);
        if (cached_nodes.contains(n))
            std::copy_n(cached_nodes.at(n).begin(), n, q._x.get());
        else
        {
            // use the Golub-Welsch algorithm
            std::fill_n(&q._x[1], n-2, 0.0);
            std::vector<double> _E(n-3);
            double * E = _E.data();
            for (int i=0; i < n-3; ++i)
            {
                double ii = i+1;
                E[i] = std::sqrt( ii * (ii + 2.0) / ((2.0*ii + 3.0) * (2.0*ii + 1.0)) );
            }

            int N = n-2;
            int info;
            char only_eigvals = 'N';
            int LDZ_dummy = 1;

            dsteqr_(&only_eigvals, &N, &q._x[1], E, nullptr, &LDZ_dummy, nullptr, &info);

            if (info != 0)
                throw std::runtime_error("gauss_lobatto() error: eigenvalue decomposition failed!");

            q._x[0] = -1.0;
            q._x[n-1] = 1.0;

            // refine roots with Newton's method
            for (int i=1; i < n/2; ++i)
            {
                for (int j=0; j < 3; ++j) // just two iterations, shouldn't need much refinement
                {
                    double P = jacobiP(n-2, 1, 1, q._x[i]);
                    double dP = jacobiP_derivative(1, n-2, 1, 1, q._x[i]);
                    q._x[i] -= P/dP;
                }

                q._x[n-1-i] = -q._x[i];
            }

            if (n & 1)
                q._x[n/2] = 0.0;
        }

        for (int i=0; i < n; ++i)
            q._w[i] = 2.0 / (n*(n-1) * square(std::legendre(n-1, q._x[i])));

        return q;
    }

    QuadratureRule QuadratureRule::gauss_legendre(int n)
    {
        if (n < 1)
            throw std::invalid_argument("gauss_legendre error: require n >= 1, but n = " + std::to_string(n) + ".");
        
        static const std::unordered_map<int, std::vector<double>> cached_nodes = {
            {1, {0.0}},
            {2, {-0.577350269189625764509149, 0.577350269189625764509149}},
            {3, {-0.774596669241483377035853, 0.0, 0.774596669241483377035853}},
            {4, {-0.861136311594052575223946, -0.339981043584856264802666, 0.339981043584856264802666, 0.861136311594052575223946}},
            {5, {-0.906179845938663992797627, -0.538469310105683091036314, 0.0, 0.538469310105683091036314, 0.906179845938663992797627}},
            {6, {-0.932469514203152027812302, -0.661209386466264513661400, -0.238619186083196908630502, 0.238619186083196908630502, 0.661209386466264513661400, 0.932469514203152027812302}},
            {7, {-0.949107912342758524526190, -0.741531185599394439863865, -0.405845151377397166906606, 0.0, 0.405845151377397166906606, 0.741531185599394439863865, 0.949107912342758524526190}},
            {8, {-0.960289856497536231683561, -0.796666477413626739591554, -0.525532409916328985817739, -0.183434642495649804939476, 0.183434642495649804939476, 0.525532409916328985817739, 0.796666477413626739591554, 0.960289856497536231683561}},
            {9, {-0.968160239507626089835576, -0.836031107326635794299430, -0.613371432700590397308702, -0.324253423403808929038538, 0.0, 0.324253423403808929038538, 0.613371432700590397308702, 0.836031107326635794299430, 0.968160239507626089835576}},
            {10, {-0.973906528517171720077964, -0.865063366688984510732097, -0.679409568299024406234327, -0.433395394129247190799266, -0.148874338981631210884826, 0.148874338981631210884826, 0.433395394129247190799266, 0.679409568299024406234327, 0.865063366688984510732097, 0.973906528517171720077964}}
        };

        QuadratureRule q(n, QuadratureRule::GaussLegendre);
        if (cached_nodes.contains(n))
            std::copy_n(cached_nodes.at(n).data(), n, q._x.get());
        else
        {
            // Golub-Welsch algorithm
            std::fill_n(q._x.get(), n, 0.0);
            double * E = new double[n-1];
            for (int i=0; i < n-1; ++i)
            {
                double k = i + 1;
                E[i] = k * std::sqrt(1.0 / (4.0 * k * k - 1.0));
            }

            int info;
            char only_eigvals = 'N';
            int LDZ_dummy = 1;

            dsteqr_(&only_eigvals, &n, q._x.get(), E, nullptr, &LDZ_dummy, nullptr, &info);
            delete[] E;

            if (info != 0)
                throw std::runtime_error("gauss_legendre() error: dsteqr() failed to compute eigenvalues of companion matrix.");

            // refine roots with Newton's method
            for (int i=0; i < n/2; ++i)
            {
                for (int j=0; j < 3; ++j) // only three iterations, roots should be good to start with
                {
                    double P = jacobiP(n, 0, 0, q._x[i]);
                    double dP = jacobiP_derivative(1, n, 0, 0, q._x[i]);
                    q._x[i] -= P/dP;
                }

                q._x[n-1-i] = -q._x[i];
            }

            if (n & 1)
                q._x[n/2] = 0.0;
        }

        for (int i=0; i < n; ++i)
            q._w[i] = 2.0 / (1.0 - square(q._x[i])) / square(jacobiP_derivative(1, n, 0, 0, q._x[i]));

        return q;
    }

    const QuadratureRule * QuadratureRule::quadrature_rule(int n, QuadratureType rule)
    {
        typedef std::unique_ptr<QuadratureRule> qptr;
        static std::unordered_map<int, qptr> legendre_rules;
        static std::unordered_map<int, qptr> lobatto_rules;

        if (rule == QuadratureRule::GaussLegendre)
        {
            if (not legendre_rules.contains(n))
                legendre_rules.insert({n, qptr(new QuadratureRule{gauss_legendre(n)})});

            return legendre_rules.at(n).get();
        }
        else
        {
            if (not lobatto_rules.contains(n))
                lobatto_rules.insert({n, qptr(new QuadratureRule{gauss_lobatto(n)})});
            
            return lobatto_rules.at(n).get();
        }
    }
} // namespace dg