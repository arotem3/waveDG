#ifndef DG_MESH_1D_HPP
#define DG_MESH_1D_HPP

#include <memory>
#include <limits>

#include "wdg_config.hpp"
#include "QuadratureRule.hpp"
#include "Tensor.hpp"
#include "Edge.hpp"

#ifdef WDG_USE_MPI
#include "Serializer.hpp"
#endif

namespace dg
{
    class Element1D
    {
    private:
        const double ab[2];

    public:
        int id;

        Element1D(const double * x);

        ~Element1D() = default;

    #ifdef WDG_USE_MPI
        void serialize(util::Serializer& serializer) const;

        Element1D(const int * data_ints, const double * data_doubles);
    #endif

        double physical_coordinates(double xi) const;

        double jacobian() const;

        const double * end_points() const;
    };

    struct Face1D
    {
    public:
        int id;
        int elements[2];
        FaceType type;

        Face1D() : id{-1}, elements{-1, -1} {}
        ~Face1D() = default;
    };

    class Mesh1D
    {
    public:
        class ElementMetricCollection
        {
        public:
            ElementMetricCollection(const Mesh1D& mesh, const QuadratureRule * quad);

            ElementMetricCollection(ElementMetricCollection&&) = default;

            const double * jacobians() const;

            const double * physical_coordinates() const;

        private:
            const Mesh1D& mesh;
            const QuadratureRule * const quad;

            mutable std::unique_ptr<double[]> J;
            mutable std::unique_ptr<double[]> x;
        };

        Mesh1D() = default;
        ~Mesh1D() = default;

        Mesh1D(const Mesh1D&) = delete;
        Mesh1D& operator=(const Mesh1D&) = delete;

        Mesh1D(Mesh1D&&) = default;
        Mesh1D& operator=(Mesh1D&&) = default;

        int n_elem() const
        {
            return _elements.size();
        }

        int n_faces(FaceType type) const
        {
            if (type == Face1D::INTERIOR)
                return _interior_faces.size();
            else // BOUNDARY
                return _boundary_faces.size();
        }

        double min_h() const;

        double max_h() const;

        const Element1D& element(int el) const
        {
        #ifdef WDG_DEBUG
            if (el < 0 || el >= (int)_elements.size())
                wdg_error("Mesh1D::element error: element index out of range.");
        #endif

            return _elements[el];
        }

        const Face1D& face(int i, FaceType type) const
        {
            if (type == Face1D::INTERIOR)
            {
            #ifdef WDG_DEBUG
                if (i < 0 || i >= _interior_faces.size())
                    wdg_error("Mesh1D::face error: face index out of range.");
            #endif
                return _interior_faces[i];
            }
            else // BOUNDARY
            {
            #ifdef WDG_DEBUG
                if (i < 0 || i >= _boundary_faces.size())
                    wdg_error("Mesh1D::face error: face index out of range.");
            #endif
                return _boundary_faces[i];
            }
        }

        const ElementMetricCollection& element_metrics(const QuadratureRule * quad) const;

        static Mesh1D from_vertices(int nx, const double * x);

        static Mesh1D uniform_mesh(int nel, double a, double b);

    #ifdef WDG_USE_MPI
        void distribute();

        int global_n_elem() const;

        int find_element(int el) const
        {
        #ifdef WDG_DEBUG
            if (el  <= 0 || el >= (int)e2p.size())
                wdg_error("Mesh1D::find_element error: element index out of range.");
        #endif

            return e2p[el];
        }
    #endif

    private:
        std::vector<Element1D> _elements;
        std::vector<Face1D> _interior_faces;
        std::vector<Face1D> _boundary_faces;

        mutable std::unordered_map<const QuadratureRule *, ElementMetricCollection> elem_collections;

    #ifdef WDG_USE_MPI
        std::vector<int> e2p;
        std::unordered_map<int, int> _elem_local_id;
    #endif
    };
} // namespace dg


#endif