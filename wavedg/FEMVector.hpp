#ifndef DG_FEM_VECTOR_HPP
#define DG_FEM_VECTOR_HPP

#include <vector>
#include <map>

#include "wdg_config.hpp"
#include "Tensor.hpp"
#include "Mesh1D.hpp"
#include "Mesh2D.hpp"

namespace dg
{
    /// @brief representation of FEM degrees of freedom restricted to element
    /// faces. This class is used for computing DG fluxes. When using MPI, this
    /// class handles communication of face values between distributed elements
    /// that share a face.
    class FaceVector
    {
    public:
        FaceVector(int n_var, const Mesh1D& mesh, FaceType face_type, const QuadratureRule * basis);
        FaceVector(int n_var, const Mesh2D& mesh, FaceType face_type, const QuadratureRule * basis);

        ~FaceVector() = default;

        /// @brief returns the size of the face vector
        inline int size() const
        {
            return x.size();
        }

        /// @brief returns the vector dimension 
        inline int n_var() const
        {
            return _n_var;
        }

        /// @brief returns the number of basis functions on each face
        inline int n_basis() const
        {
            return _n_basis;
        }

        /// @brief returns the number of faces 
        inline int n_faces() const
        {
            return _n_faces;
        }

        /// @brief returns the face type 
        inline FaceType face_type() const
        {
            return _face_type;
        }

        /// @brief returns the face vector DOFs. Shape (n_basis, n_var, 2, n_faces) 
        double * get()
        {
            return x;
        }

        /// @brief returns the face vector DOFs. Shape (n_basis, n_var, 2, n_faces) 
        const double * get() const
        {
            return x;
        }
        
        /// @brief returns the face vector DOFs. Shape (n_basis, n_var, 2, n_faces) 
        operator double*()
        {
            return x;
        }

        /// @brief returns the face vector DOFs. Shape (n_basis, n_var, 2, n_faces) 
        operator const double*() const
        {
            return x;
        }

        /// @brief returns the face vector DOFs as a tensor of shape (n_basis, n_var, 2, n_faces) 
        inline TensorWrapper<4, double> as_tensor()
        {
            return x;
        }

        /// @brief returns the face vector DOFs as a tensor of shape (n_basis, n_var, 2, n_faces) 
        inline TensorWrapper<4, const double> as_tensor() const
        {
            return reshape(x.data(), _n_basis, _n_var, 2, _n_faces);
        }

        /// @brief returns the face vector DOFs as a vector 
        inline dvec_wrapper as_dvec()
        {
            return reshape(x, size());
        }

        /// @brief returns the face vector DOFs as a vector 
        inline const_dvec_wrapper as_dvec() const
        {
            return reshape(x.data(), size());
        }

        /// @brief returns a pointer to the face DOFs of face @a f 
        inline double * face(int f)
        {
            return &x(0, 0, 0, f);
        }

        /// @brief returns a pointer to the face DOFs of face @a f 
        inline const double * face(int f) const
        {
            return &x(0, 0, 0, f);
        }

        /// @brief returns a pointer to the face DOFs of side @a s on face @a f 
        inline double * side(int s, int f)
        {
            return &x(0, 0, s, f);
        }

        /// @brief returns a pointer to the face DOFs of side @a s on face @a f 
        inline const double * side(int s, int f) const
        {
            return &x(0, 0, s, f);
        }

        /// @brief access face dof (i, d, s, f) 
        inline double& operator()(int i, int d, int s, int f)
        {
            return x(i, d, s, f);
        }

        /// @brief access face dof (i, d, s, f) 
        inline const double& operator()(int i, int d, int s, int f) const
        {
            return x(i, d, s, f);
        }

    #ifdef WDG_USE_MPI
        /// @brief distributes face values between processors that share faces
        void send_recv() const;
    #endif

    private:
        const int dim;
        const int _n_var;
        const int _n_basis;
        const int _n_faces;

        const FaceType _face_type;

        Tensor<4, double> x;

    #ifdef WDG_USE_MPI
        struct PersistantChannel
        {
            int partner; // rank to send/recv with
            ivec faces_to_send; // indices of edges to send
            ivec faces_to_recv; // indices of edges to recv
            mutable dvec send_buf;
            mutable dvec recv_buf;
        };

        std::vector<PersistantChannel> channels;
        mutable RequestVec rreq;
        mutable RequestVec sreq;
    #endif
    };

    /// @brief representation of FEM degrees of freedom
    class FEMVector
    {
    public:
        FEMVector(int n_var, const Mesh1D& mesh, const QuadratureRule * basis);
        FEMVector(int n_var, const Mesh2D& mesh, const QuadratureRule * basis);

        ~FEMVector() = default;

        /// @brief returns the degrees of freedom (on this processor). 
        inline int size() const
        {
            return x.size();
        }

        /// @brief returns the vector dimension of FEMVector 
        inline int n_var() const
        {
            return _n_var;
        }

        /// @brief returns the number of basis functions on each element. 
        inline int n_basis() const
        {
            return _n_basis;
        }

        /// @brief returns the number of elements. 
        inline int n_elem() const
        {
            return _n_elem;
        }

        /// @brief returns the vector of degrees of freedom. Shape (n_var, n_basis, n_elem)
        inline double * get()
        {
            return x;
        }

        /// @brief returns the vector of degrees of freedom. Shape (n_var, n_basis, n_elem)
        inline const double * get() const
        {
            return x;
        }

        inline operator double *()
        {
            return x;
        }

        inline operator const double * () const
        {
            return x;
        }

        /// @brief returns the degrees of freedom as a tensor of shape (n_var, n_basis, n_elem)
        inline dcube_wrapper as_tensor()
        {
            return x;
        }

        /// @brief returns the degrees of freedom as a tensor of shape (n_var, n_basis, n_elem)
        inline const_dcube_wrapper as_tensor() const
        {
            return const_dcube_wrapper(x.data(), _n_var, _n_basis, _n_elem);
        }

        /// @brief returns the DOFs as a vector 
        inline dvec_wrapper as_dvec()
        {
            return reshape(x, size());
        }

        /// @brief returns the DOFs as a vector 
        inline const_dvec_wrapper as_dvec() const
        {
            return reshape(x.data(), size());
        }

        /// @brief returns a pointer to the degrees of freedom of element @a el.
        inline double * element(int el)
        {
            return &x(0, 0, el);
        }

        /// @brief returns a pointer to the degrees of freedom of element @a el.
        inline const double * element(int el) const
        {
            return &x(0, 0, el);
        }

        /// @brief returns a pointer to the degrees of freedom of basis function @a i on element @a el. 
        inline double * basis(int el, int i)
        {
            return &x(0, i, el);
        }

        /// @brief returns a pointer to the degrees of freedom of basis function @a i on element @a el. 
        inline const double * basis(int el, int i) const
        {
            return &x(0, i, el);
        }

        /// @brief access DOF (d, i, el)
        inline double& operator()(int d, int i, int el)
        {
            return x(d, i, el);
        }

        /// @brief access DOF (d, i, el) 
        inline const double& operator()(int d, int i, int el) const
        {
            return x(d, i, el);
        }

    private:
        const int dim;
        const int _n_var;
        const int _n_basis;
        const int _n_elem;

        dcube x;
    };
} // namespace dg

#endif