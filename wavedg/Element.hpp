#ifndef DG_ELEMENT_HPP
#define DG_ELEMENT_HPP

#include "config.hpp"
#include "Tensor.hpp"

namespace dg
{
    class Element
    {
    public:
        int id;

        // maps the reference coordinates xi to the physical coordinates x
        virtual void physical_coordinates(const double * xi, double * x) const = 0;
        
        // computes the jacobian of the element mapping from reference coordinates
        // to physical coordinates. J(i, j) = dx[j]/dxi[i] so that dx = J*dxi where
        // x is the physical coordinate and xi is the reference coordinate.
        virtual void jacobian(const double * xi, double * J) const = 0;

        // computes the determinant of the jacobian of element mapping from
        // reference coordinates to physical coordinates.
        virtual double measure(const double * xi) const
        {
            double J[4];
            jacobian(xi, J);
            return J[0]*J[3] - J[1]*J[2];
        }
    };

    class QuadElement : public Element
    {
    private:
        double x[4][2];

    public:
        void physical_coordinates(const double * xi, double * x) const override;
        void jacobian(const double * xi, double * J) const override;

        /// @brief initialize QuadElement by providing the coordinates of its
        /// corners (in counter clockwise order)
        /// @param xs share (2, 4) so that xs(0, i) = x[i] and xs(1, i) = y[i].
        QuadElement(const double * xs);

        // returns the coordinates of the i-th corner.
        inline const double * corner(int i) const
        {
            return x[i];
        }
    };
} // namespace dg

#endif