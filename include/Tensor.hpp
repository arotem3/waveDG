#ifndef DG_TENSOR_HPP
#define DG_TENSOR_HPP

#include <vector>
#include <stdexcept>
#include <memory>

namespace dg
{
    template <typename Size, typename... Sizes>
    inline int tensor_dims(int * dim, Size s, Sizes... sizes)
    {
        if (s < 0)
            throw std::logic_error("tensor cannot have negative dimension");
        
        *dim = s;
        if constexpr (sizeof...(sizes) > 0)
        {
            ++dim;
            return s * tensor_dims(dim, sizes...);
        }
        else
            return s;
    }

    template <typename Ind, typename... Inds>
    inline int tensor_index(const int * dims, Ind idx, Inds... ids)
    {
        #ifdef DG_DEBUG
        if (idx < 0 || idx >= *dims)
            throw std::out_of_range("tensor index out of range.");
        #endif

        if constexpr (sizeof...(ids) > 0)
        {
            const int dim = *dims;
            ++dims;
            return idx + dim * tensor_index(dims, ids...);
        }
        else
            return idx;
    }

    template <int Dim, typename scalar>
    class TensorWrapper
    {
    protected:
        int sizes[Dim];
        int len;
        scalar * ptr;
    
    public:
        TensorWrapper() : sizes{0}, len{0}, ptr(nullptr) {};
        TensorWrapper(const TensorWrapper&) = default;
        TensorWrapper& operator=(const TensorWrapper&) = default;

        template <typename... Sizes>
        inline explicit TensorWrapper(scalar * data_, Sizes... sizes_) : ptr(data_)
        {
            static_assert(Dim > 0, "Tensor must have a positive number of dimensions");
            static_assert(sizeof...(sizes_) == Dim, "Wrong number of dimensions specified.");
            
            len = tensor_dims(sizes, sizes_...);
        }

        template <typename... Indices>
        inline scalar& operator()(Indices... ids)
        {
            static_assert(sizeof...(ids) == Dim, "Wrong number of indices specified.");

            #ifdef DG_DEBUG
            if (ptr == nullptr)
                throw std::runtime_error("TensorWrapper memory uninitialized.");
            #endif

            return ptr[tensor_index(sizes, ids...)];
        }

        template <typename... Indices>
        inline const scalar& operator()(Indices... ids) const
        {
            static_assert(sizeof...(ids) == Dim, "Wrong number of indices specified.");

            #ifdef DG_DEBUG
            if (ptr == nullptr)
                throw std::runtime_error("TensorWrapper memory uninitialized.");
            #endif

            return ptr[tensor_index(sizes, ids...)];
        }

        inline scalar& operator[](int idx)
        {
            #ifdef DG_DEBUG
            if (ptr == nullptr)
                throw std::runtime_error("TensorWrapper memory uninitialized.");
            if (idx < 0 || idx >= len)
                throw std::out_of_range("tensor linear index out of bounds");
            #endif

            return ptr[idx];
        }

        inline const scalar& operator()(int idx) const
        {
            #ifdef DG_DEBUG
            if (ptr == nullptr)
                throw std::runtime_error("TensorWrapper memory uninitialized.");
            if (idx < 0 || idx >= len)
                throw std::out_of_range("tensor linear index out of bounds");
            #endif

            return ptr[idx];
        }
    
        inline scalar * data()
        {
            return ptr;
        }

        inline const scalar * data() const
        {
            return ptr;
        }
    
        inline const int * shape() const
        {
            return sizes;
        }    
    
        inline int size() const
        {
            return len;
        }
    };

    template <typename scalar, typename... Sizes>
    inline TensorWrapper<sizeof...(Sizes), scalar> reshape(scalar * data, Sizes... shape)
    {
        return TensorWrapper<sizeof...(Sizes), scalar>(data, shape...);
    }

    template <typename scalar>
    using VectorWrapper = TensorWrapper<1, scalar>;
    template <typename scalar>
    using MatrixWrapper = TensorWrapper<2, scalar>;
    template <typename scalar>
    using CubeWrapper = TensorWrapper<3, scalar>;

    typedef TensorWrapper<1, double> dvec_wrapper;
    typedef TensorWrapper<1, const double> const_dvec_wrapper;

    typedef TensorWrapper<2, double> dmat_wrapper;
    typedef TensorWrapper<2, const double> const_dmat_wrapper;

    typedef TensorWrapper<3, double> dcube_wrapper;
    typedef TensorWrapper<3, const double> const_dcube_wrapper;

    template <int Dim, typename scalar>
    class Tensor : public TensorWrapper<Dim, scalar>
    {
    private:
        std::unique_ptr<scalar[]> mem;

    public:
        inline Tensor() : TensorWrapper<Dim, scalar>() {}
        Tensor(const Tensor&) = delete;
        Tensor& operator=(const Tensor&) = delete;
        Tensor(Tensor&&) = default;
        Tensor& operator=(Tensor&&) = default;

        template <typename... Sizes>
        inline explicit Tensor(Sizes... sizes_) : TensorWrapper<Dim, scalar>(nullptr, sizes_...), mem(new scalar[this->len])
        {
            this->ptr = mem.get();
        }
    };

    template <typename scalar>
    using Vec = Tensor<1, scalar>;
    template <typename scalar>
    using Matrix = Tensor<2, scalar>;
    template <typename scalar>
    using Cube = Tensor<3, scalar>;

    typedef Vec<double> dvec;
    typedef Matrix<double> dmat;
    typedef Cube<double> dcube;
} // namespace dg

#endif