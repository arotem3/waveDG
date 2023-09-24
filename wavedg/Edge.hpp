#ifndef DG_EDGE_HPP
#define DG_EDGE_HPP

#include "config.hpp"
#include <cmath>

namespace dg
{
    enum EdgeType
    {
        INTERIOR,
        BOUNDARY
    };

    struct Edge
    {
    public:
        EdgeType type;
        int id;
        int elements[2];
        int sides[2];
        int delta;
    
        virtual void normal(const double xi, double * n) const = 0;
        virtual double measure(const double xi) const = 0;
        virtual void physical_coordinates(const double xi, double * x) const = 0;

        Edge() : id{-1}, elements{-1, -1}, sides{-1, -1} {}
    };

    struct StraightEdge : public Edge
    {
    private:
        double n[2];
        double meas;
        double x[2];
        double dx[2];

    public:

        StraightEdge(const double * x0, const double * x1, int side)
        {
            x[0] = x0[0];
            x[1] = x0[1];

            dx[0] = x1[0] - x0[0];
            dx[1] = x1[1] - x0[1];

            const double s = std::hypot(dx[0], dx[1]);
            const double sgn = (side == 2 || side == 3) ? -1 : 1;

            n[0] = sgn * dx[1] / s;
            n[1] = -sgn * dx[0] / s;

            meas = s / 2;
        }

        inline void normal(const double xi, double * n_) const override
        {
            n_[0] = n[0];
            n_[1] = n[1];
        }

        inline double measure(const double) const override
        {
            return meas;
        }
    
        inline void physical_coordinates(const double xi, double * x_) const override
        {
            const double t = 0.5 * (xi + 1.0);

            x_[0] = x[0] + dx[0] * t;
            x_[1] = x[1] + dx[1] * t;
        }
    };
} // namespace dg


#endif