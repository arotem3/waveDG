#ifndef ODE_HPP
#define ODE_HPP

#include <vector>
#include <stdexcept>

namespace ode
{
    // Explicit five stage fourth order low storage Runge Kutta method. Has 2*n
    // storage cost.
    // see:
    //"Fourth-order 2N-storage Runge-Kutta schemes" M.H. Carpenter and C. Kennedy
    class RungeKutta4
    {
    private:
        const int n;
        mutable std::vector<double> p;
        mutable std::vector<double> du;

    public:
        // n is the length of the vectors to integrate. RungeKutta4 allocates
        // 2*n*sizeof(double) in memory.
        inline explicit RungeKutta4(int n_) : n(n_), p(n), du(n) {}
       
        /// @brief integrates ode u'(t) = F(t, u(t)) from t to t+dt.
        /// @tparam Func invocable with handle void(double*, double, const T*)
        /// @param dt the time step
        /// @param f invocable as f(double* F, double t, const double* u). On exit, F =
        /// F(t, u). f is invoked five times during each call to step.      
        /// @param t On entry, t is the initial time. On exit, t <- t+dt.
        /// @param u On entry, u = u(t) is the initial value. On exit, u <- u(t+dt).
        template <typename Func>
        void step(double dt, Func f, double& t, double * u) const
        {
            constexpr double rk4a[5] = {0.0, -567301805773.0/1357537059087.0, -2404267990393.0/2016746695238.0, -3550918686646.0/2091501179385.0, -1275806237668.0/842570457699.0};
            constexpr double rk4b[5] = {1432997174477.0/9575080441755.0, 5161836677717.0/13612068292357.0, 1720146321549.0/2090206949498.0, 3134564353537.0/4481467310338.0, 2277821191437.0/14882151754819.0};
            constexpr double rk4c[5] = {0.0, 1432997174477.0/9575080441755.0, 2526269341429.0/6820363962896.0, 2006345519317.0/3224310063776.0, 2802321613138.0/2924317926251.0};

            f(p.data(), t, u);

            for (int i=0; i < n; ++i)
            {
                du[i] = dt * p[i];
                u[i] += rk4b[0] * du[i];
            }

            for (int stage=1; stage < 5; ++stage)
            {
                const double s = t + rk4c[stage] * dt;
                f(p.data(), s, u);

                for (int i=0; i < n; ++i)
                {
                    du[i] = rk4a[stage] * du[i] + dt * p[i];
                    u[i] += rk4b[stage] * du[i];
                }
            }

            t += dt;
        }
    };

    // Explicit midpoint method (second order). Has 2*n storage cost.
    // see:
    // https://en.wikipedia.org/wiki/Midpoint_method
    class RungeKutta2
    {
    private:
        const int n;
        mutable std::vector<double> p;
        mutable std::vector<double> y;
    
    public:
        // n is the length of the vectors to integrate. RungeKutta2 allocates
        // 2*n*sizeof(double) memory.
        inline explicit RungeKutta2(int n_) : n(n_), p(n), y(n) {}

        /// @brief integrates ode u'(t) = F(t, u(t)) from t to t+dt.
        /// @tparam Func invocable with handle void(double *, double, const double *)
        /// @param dt the time step
        /// @param f invocable as f(double * F, double t, const double * u). On exit, F =
        /// F(t, u). f is invoked twice during each call to step.
        /// @param t On entry, t is the initial time. On exit, t <- t+dt.
        /// @param u On entry, u = u(t) is the initial value. On exit, u <- u(t+dt).
        template <typename Func>
        void step(double dt, Func f, double& t, double * u) const
        {
            f(p.data(), t, u);

            const double half_dt = 0.5 * dt;
            for (int i=0; i < n; ++i)
                y[i] = u[i] + half_dt * p[i];

            const double s = t + half_dt;
            f(p.data(), s, y.data());

            for (int i=0; i < n; ++i)
                u[i] += dt * p[i];

            t += dt;
        }
    };

    // Explicit three stage third order stability preserving Runge Kutta method.
    // Method has 2*n storage cost.
    // see:
    // "Strong stability preserving high order time discretization methods"
    // S. Gottlieb, C.-W. Shu, and E. Tadmor
    class SSPRK3
    {
    private:
        const int n;
        mutable std::vector<double> p;
        mutable std::vector<double> y;
    
    public:
        // n is the length of the vectors to integrate. SSPRK3 allocates
        // 2*n*sizeof(double) memory.
        inline explicit SSPRK3(int n_) : n(n_), p(n), y(n) {}

        /// @brief integrates ode u'(t) = F(t, u(t)) from t to t+dt.
        /// @tparam Func invocable with handle void(T*, double, const T*)
        /// @param dt the time step
        /// @param f invocable as f(T* F, double t, const T* u). On exit, F =
        /// F(t, u). f is invoked three times during each call to step.
        /// @param t On entry, t is the initial time. On exit, t <- t+dt.
        /// @param u On entry, u = u(t) is the initial value. On exit, u <- u(t+dt).
        template <typename Func>
        void step(double dt, Func f, double& t, double * u) const
        {
            f(p.data(), t, u);

            for (int i=0; i < n; ++i)
                y[i] = u[i] + dt * p[i];
            
            double s = t + dt;
            f(p.data(), s, y.data());

            for (int i=0; i < n; ++i)
                y[i] = 0.25 * (y[i] + 3.0 * u[i] + dt * p[i]);

            s = t + 0.5 * dt;
            f(p.data(), s, y.data());

            for (int i=0; i < n; ++i)
                u[i] = (u[i] + 2.0 * y[i] + 2.0 * dt * p[i]) / 3.0;

            t += dt;
        }
    };
} // namespace ode


#endif