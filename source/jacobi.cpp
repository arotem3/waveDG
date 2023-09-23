#include "jacobi.hpp"

namespace dg
{
    double jacobiP_next(unsigned int m, double a, double b, double x, double y1, double y2)
    {
        double yp1 = (2*m + a + b - 1)*((2*m + a + b)*(2*m + a + b - 2)*x + a*a - b*b)*y1
                        - 2*(m + a - 1)*(m + b - 1)*(2*m + a + b)*y2;
        yp1 /= 2*m*(m + a + b)*(2*m + a + b - 2);
        return yp1;
    }

    double jacobiP(unsigned int n, double a, double b, double x)
    {
        double ym1 = 1;
        
        if (n == 0)
            return ym1;
        
        double y = (a + 1) + 0.5*(a + b + 2)*(x-1);
        
        for (unsigned int m=2; m <= n; ++m)
        {
            double yp1 = jacobiP_next(m, a, b, x, y, ym1);
            ym1 = y;
            y = yp1;
        }

        return y;
    }

    double jacobiP_derivative(unsigned int k, unsigned int n, double a, double b, double x)
    {
        if (k > n)
            return 0.0;
        else
        {
            double s = std::lgamma(n+a+b+1+k) - std::lgamma(n+a+b+1) - k * std::log(2);
            return std::exp(s) * jacobiP(n-k, a+k, b+k, x);
        }
    }
} // namespace dg