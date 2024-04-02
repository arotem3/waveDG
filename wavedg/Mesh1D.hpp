#ifndef DG_MESH_1D_HPP
#define DG_MESH_1D_HPP

#include <memory>
#include <limits>
#include <vector>
#include <set>

#include "wdg_config.hpp"
#include "QuadratureRule.hpp"
#include "Tensor.hpp"
#include "Edge.hpp"

#ifdef WDG_USE_MPI
#include "Serializer.hpp"
#endif

namespace dg
{
    /// @brief Represents an element of a one dimensional mesh (i.e. an interval).
    class Element1D
    {
    private:
        const double ab[2];

    public:
        int id; ///< the global index of the element in the mesh. Typically defined by the mesh.

        /// @brief initialize 1D element by specifying end-points
        /// @param x The left and right end points of the interval
        Element1D(const double x[2]);

        ~Element1D() = default;

    #ifdef WDG_USE_MPI
        void serialize(util::Serializer& serializer) const;

        Element1D(const int * data_ints, const double * data_doubles);
    #endif

        /// @brief computes the physical coordinate of the reference variable xi
        /// @param xi the coordinate in the reference interval [-1, 1]
        /// @return the physical coordinate
        inline double physical_coordinates(double xi) const
        {
            return ab[0] + 0.5 * (ab[1] - ab[0]) * (xi + 1.0);
        }

        /// @brief computes the Jacobian of the transformation from reference to
        /// physical coordinates: \f$\frac{dx}{d\xi}\f$.
        ///
        /// @details In 1D this is just the const h/2 where h is the length of
        /// the interval.
        /// @return The Jacobian: \f$\frac{dx}{d\xi}\f$.
        inline double jacobian() const
        {
            return 0.5 * (ab[1] - ab[0]);
        }

        /// @brief returns a pointer to the left and right end-points of the interval.
        inline const double * end_points() const
        {
            return ab;
        }
    };

    /// @brief represents a face between two one dimensional elements.
    struct Face1D
    {
    public:
        int id; ///< global index of the face in the mesh. Typically defined by the mesh.
        int elements[2]; ///< global indices of the elements to the left and right of the face.
        FaceType type; ///< specifies if the face is an INTERIOR face or a BOUNDARY face.

        /// @brief initializes a 1D face.
        ///
        /// @details The default values of id and elements are -1 which is an
        /// invalid index and therefore has to be specified after construction.
        Face1D() : id{-1}, elements{-1, -1} {}

    #ifdef WDG_USE_MPI
        void serialize(util::Serializer& serializer) const;

        Face1D(const int * data_ints, const double * data_doubles);
    #endif
        
        ~Face1D() = default;
    };

    /// @brief A one dimensional mesh of Element1D and Face1D.
    class Mesh1D
    {
    public:
        /// @brief Manages metrics (evaluated on a specific collocation set) for
        /// 1D elements in a mesh from which this ElementMetricCollection was
        /// constructed.
        class ElementMetricCollection
        {
        public:
            /// @brief initializes metric collection for @a mesh on collocation rule @a quad
            /// @param mesh The 1D mesh
            /// @param quad The collocation rule to evaluate metrics on.
            /// (pointer must be valid for the life time of this object)
            ElementMetricCollection(const Mesh1D& mesh, const QuadratureRule * quad);

            /// @brief move ElementMetricCollection
            ElementMetricCollection(ElementMetricCollection&&) = default;

            /// @brief returns the Jacobian (of the transformation from
            /// reference to physical) on the collocation rule @a quad for every
            /// element.
            ///
            /// @details The first time `jacobians()` is called, the array of
            /// Jacobian values is computed and stored internally, so on every
            /// subsequent call to `jacobians()`, the array does not need to be
            /// recomputed.
            ///
            /// @return Shape (n_quad, n_elem).
            const double * jacobians() const;

            /// @brief returns the physical coordinates on the collocation rule
            /// @a quad for every element.
            ///
            /// @details The first time `physical_coordinates()` is called, the
            /// array of coordinates is computed and stored internally, so on
            /// every subsequent call to `physical_coordinates()`, the array
            /// does not need to be recomputed.
            ///
            /// @return Shape (n_quad, n_elem).
            const double * physical_coordinates() const;

        private:
            const Mesh1D& mesh;
            const QuadratureRule * const quad;

            mutable std::unique_ptr<double[]> J;
            mutable std::unique_ptr<double[]> x;
        };

        /// @brief empty mesh
        Mesh1D() = default;

        ~Mesh1D() = default;

        /// @brief copy mesh
        Mesh1D(const Mesh1D&) = delete;

        /// @brief copy mesh
        Mesh1D& operator=(const Mesh1D&) = delete;

        /// @brief move mesh 
        Mesh1D(Mesh1D&&) = default;

        /// @brief move mesh
        Mesh1D& operator=(Mesh1D&&) = default;

        /// @brief returns the number of elements in the mesh. 
        int n_elem() const
        {
            return _elements.size();
        }

        /// @brief returns the number of faces (of FaceType type) in the mesh. 
        int n_faces(FaceType type) const
        {
            if (type == FaceType::INTERIOR)
                return _interior_faces.size();
            else // BOUNDARY
                return _boundary_faces.size();
        }

        /// @brief returns the smallest element length.
        double min_h() const;

        /// @brief returns the largets element length.
        double max_h() const;

        /// @brief returns a reference to Element1D @a el in the mesh.
        const Element1D& element(int el) const
        {
        #ifdef WDG_DEBUG
            if (el < 0 || el >= (int)_elements.size())
                wdg_error("Mesh1D::element error: element index out of range.");
        #endif

            return _elements[el];
        }

        /// @brief returns a reference to Face1D @a i of FaceType @a type in the mesh.
        const Face1D& face(int i, FaceType type) const
        {
            if (type == FaceType::INTERIOR)
            {
            #ifdef WDG_DEBUG
                if (i < 0 || (unsigned long)i >= _interior_faces.size())
                    wdg_error("Mesh1D::face error: face index out of range.");
            #endif
                return _faces[_interior_faces[i]];
            }
            else // BOUNDARY
            {
            #ifdef WDG_DEBUG
                if (i < 0 || (unsigned long)i >= _boundary_faces.size())
                    wdg_error("Mesh1D::face error: face index out of range.");
            #endif
                return _faces[_boundary_faces[i]];
            }
        }

        /// @brief returns a metric collection for the specified quadrature
        /// rule. From this collection, the jacobians and physical coordiantes
        /// on the points of the quadrature rule on each element can be computed.
        const ElementMetricCollection& element_metrics(const QuadratureRule * quad) const;

        /// @brief construct a 1D mesh on a sequence of vertices.
        /// @param nx The number of vertices.
        /// @param x The coordinates of the vertices. Array of Length nx. Must be sorted.
        /// @return a 1D mesh with nx-1 elements such that the end points of element el are x[el] and x[el+1].
        static Mesh1D from_vertices(int nx, const double * x, bool periodic=false);

        /// @brief construct a uniform 1D mesh on the interval [a, b]
        /// @param nel the desired number of elements
        /// @param a the left end-point of the mesh
        /// @param b the right end-point of the mesh
        /// @return a 1D mesh consisting of nel uniform elements on [a, b]
        static Mesh1D uniform_mesh(int nel, double a, double b, bool periodic=false);

    #ifdef WDG_USE_MPI
        /// @brief distribures 1D mesh from root (one which it was constructed)
        /// to the rest of the processors on MPI_COMM_WORLD in a load balanced
        /// split.
        void distribute();

        /// @brief returns the total number of elements on all processors of
        /// MPI_COMM_WORLD. This function is blocking. 
        int global_n_elem() const;

        /// @brief returns the rank of the processor which manages the element
        /// with global index @a el .
        int find_element(int el) const
        {
        #ifdef WDG_DEBUG
            if (el  < 0 || el >= (int)e2p.size())
                wdg_error("Mesh1D::find_element error: element index out of range.");
        #endif

            return e2p[el];
        }

        int local_element_index(int global_index) const
        {
        #ifdef WDG_DEBUG
            if (not _elem_local_id.contains(global_index))
                wdg_error("Mesh2D::local_element_index error: Element does not belong to this processor.");
        #endif
            return _elem_local_id.at(global_index);
        }
    
        /// @brief returns a list with elements `face_pattern[i] = s + 2*f` indicating that side `s` of face `f` is on this processor.
        inline const_ivec_wrapper face_pattern(FaceType face_type) const
        {
            if (face_type == FaceType::INTERIOR)
                return const_ivec_wrapper(interior_face_pattern.data(), interior_face_pattern.size());
            else // BOUNDARY
                return const_ivec_wrapper(boundary_face_pattern.data(), boundary_face_pattern.size());
        }
    #endif

    private:
        std::vector<Element1D> _elements;
        std::vector<Face1D> _faces;
        std::vector<int> _interior_faces;
        std::vector<int> _boundary_faces;

        mutable std::unordered_map<const QuadratureRule *, ElementMetricCollection> elem_collections;

    #ifdef WDG_USE_MPI
        std::vector<int> e2p;
        std::unordered_map<int, int> _elem_local_id;
        std::unordered_map<int, int> _face_local_id;
        mutable ivec interior_face_pattern;
        mutable ivec boundary_face_pattern;

        void compute_face_pattern() const;
    #endif
    };
} // namespace dg


#endif