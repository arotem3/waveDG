#ifndef DG_TENSOR_HPP
#define DG_TENSOR_HPP

#include "TensorView/TensorView.hpp"

using namespace tensor;

namespace dg
{
    using ivec = Vector<int>;
    using dvec = Vector<double>;
    using dmat = Matrix<double>;
    using dcube = Cube<double>;

    using dvec_wrapper = vector_view<double>;
    using dmat_wrapper = matrix_view<double>;
    using dcube_wrapper = cube_view<double>;

    using const_ivec_wrapper = vector_view<const int>;
    using const_dvec_wrapper = vector_view<const double>;
    using const_dmat_wrapper = matrix_view<const double>;
    using const_dcube_wrapper = cube_view<const double>;

    template <typename T, index_t Dim>
    void fill(TensorView<T, Dim> &x, T value)
    {
        for (auto &y : x)
            y = value;
    }

    template <typename T, index_t Dim>
    void fill(Tensor<T, Dim> &x, T value)
    {
        for (auto &y : x)
            y = value;
    }

    template <typename T, index_t Dim>
    void zeros(TensorView<T, Dim> &x)
    {
        fill(x, T(0));
    }

    template <typename T, index_t Dim>
    void zeros(Tensor<T, Dim> &x)
    {
        fill(x, T(0));
    }
}

#endif