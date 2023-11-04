#ifndef DG_MESH2D_HPP
#define DG_MESH2D_HPP

#include <vector>
#include <cmath>
#include <unordered_map>
#include <memory>
#include <fstream>

#include "wdg_config.hpp"
#include "QuadratureRule.hpp"
#include "Tensor.hpp"
#include "Element.hpp"
#include "Edge.hpp"

namespace dg
{
    /// @brief The 2D mesh.
    ///
    /// The arrays returned by element_jacobian, ... are maintained by the mesh,
    /// and will persist while the mesh exists. Since various other classes will
    /// call these member functions, it is recommended that the quadrature rule
    /// used is provided by the `quadrature_rule` function which maintains a
    /// list quadrature rules for the lifetime of the program. This ensures that
    /// the mesh computes the metrics only once for a given quadrature rule.
    class Mesh2D
    {
    private:
        std::vector<std::unique_ptr<Edge>> _edges;
        std::vector<std::unique_ptr<Element>> _elements;
        std::vector<int> _boundary_edges;
        std::vector<int> _interior_edges;

        typedef std::unordered_map<const QuadratureRule *, std::unique_ptr<double[]>> MetricCollection;
        mutable MetricCollection J;
        mutable MetricCollection detJ;
        mutable MetricCollection x;

        mutable MetricCollection n;
        mutable MetricCollection edge_x;
        mutable MetricCollection edge_meas;

        mutable MetricCollection n_int;
        mutable MetricCollection edge_x_int;
        mutable MetricCollection edge_meas_int;
        
        mutable MetricCollection n_ext;
        mutable MetricCollection edge_x_ext;
        mutable MetricCollection edge_meas_ext;

    public:
        /// @brief constructs empty mesh
    Mesh2D() {}

        /// @brief DELETED: mesh maintains unique pointers to abstract types. Copies are non-trivial.
        Mesh2D(const Mesh2D &mesh) = delete;

        /// @brief move mesh
        Mesh2D(Mesh2D &&) = default;

        /// @brief DELETED: mesh maintains unique pointers to abstract types. Copies are non-trivial. 
        Mesh2D &operator=(const Mesh2D &) = delete;

        /// @brief move mesh 
        Mesh2D &operator=(Mesh2D &&) = default;

    int n_elem() const
    {
        return _elements.size();
    }

        /// number of edges in mesh. If mesh is distributed (MPI), then returns
        /// the number of edges on this processor.
    int n_edges() const
    {
        return _edges.size();
    }

        /// number of edges of specified type in mesh. If mesh is distributed
        /// (MPI), then returns the number of edges on this processor.
        int n_edges(Edge::EdgeType type) const
    {
            if (type == Edge::BOUNDARY)
        {
            return _boundary_edges.size();
        }
        else
        {
            return _interior_edges.size();
        }
    }

        /// @brief returns the maximum polynomial degree of all element mappings.
        int max_element_order() const
        {
            // only bilinear elements supported so far
            return 1;
        }

        /// @brief returns the minimum polynomial degree of all element mappings. 
        int min_element_order() const
        {
            // only bilinear elements supported so far
            return 1;
        }

        /// @brief returns the area of the smallest element
        double min_element_measure() const;
        
        /// @brief returns the area of the largest element 
        double max_element_measure() const;

        /// @brief returns the length of the shortest edge
        double min_edge_measure() const;

        /// @brief returns the length of the longest edge 
        double max_edge_measure() const;

        /// returns the edge specified by edge index i. For distributed meshes:
        /// this index is local to the processor and should be in the range [0,
        /// n_edges() ).
        const Edge *edge(int i) const
    {
        #ifdef WDG_DEBUG
        if (i < 0 || i >= (int)_edges.size())
            throw std::out_of_range("edge index out of range");
        #endif

        return _edges[i].get();
    }

        /// returns the edge of Edge::EdgeType type specified by edge index i. For
        /// distributed meshes: this index is local to the processor and should
        /// be in the range [0, n_edges(type) ).
        const Edge *edge(int i, Edge::EdgeType type) const
    {
            if (type == Edge::BOUNDARY)
        {
            #ifdef WDG_DEBUG
            if (i < 0 || i >= (int)_boundary_edges.size())
                throw std::out_of_range("boundary edge index out of range.");
            #endif

            return _edges[_boundary_edges[i]].get();
        }
        else
        {
            #ifdef WDG_DEBUG
            if (i < 0 || i >= (int)_interior_edges.size())
                throw std::out_of_range("interior edge index out of range.");
            #endif

            return _edges[_interior_edges[i]].get();
        }
    }

        /// returns the element specified by element index el. For distributed
        /// meshes: this index is local to the processor and should be in the
        /// range [0, n_elem() ).
        const Element *element(int el) const
    {
        #ifdef WDG_DEBUG
        if (el < 0 || el >= (int)_elements.size())
            throw std::out_of_range("element index out of range.");
        #endif

        return _elements[el].get();
    }

        /// returns an array of the element jacobians evaluated on a quadrature rule.
        /// The output J has shape (2, 2, n, n, n_elem) where n is the length of the
        /// quadrature rule.
        const double *element_jacobians(const QuadratureRule *) const;

        /// returns an array of the element measures ie the determinant of the
        /// jacobians evaluated on a quadrature rule. The output detJ has shape (n,
        /// n, n_elem) where n is the length of the quadrature rule.
        const double *element_measures(const QuadratureRule *) const;
    
        /// returns an array of the physical coordinates of the quadrature rule on
        /// every element. The output x has shape (2, n, n, n_elem) where n is the
        /// length of the quadrature rule.
        const double *element_physical_coordinates(const QuadratureRule *) const;

        /// returns an array of the normal derivatives of all of the edges evaluated
        /// on the quadrature rule. The output has shape (2, n, n_edges) where n is
        /// the length of the quadrature rule.
        const double *edge_normals(const QuadratureRule *) const;

        /// returns an array of the normal derivatives of all of the edges of the
        /// requested Edge::EdgeType evaluated on the quadrature rule. The output has shape
        /// (2, n, n_edges) where n is the length of the quadrature rule.
        const double *edge_normals(const QuadratureRule *, Edge::EdgeType) const;

        /// returns an array of the physical coordinates of the quadrature rule on
        /// every edge. The output has shape (2, n, n_edges) where n is the length of
        /// the quadrature rule.
        const double *edge_physical_coordinates(const QuadratureRule *) const;

        /// returns an array of the physical coordinates of the quadrature rule on
        /// every edge of the requested type. The output has shape (2, n, n_edges)
        /// where n is the length of the quadrature rule.
        const double *edge_physical_coordinates(const QuadratureRule *, Edge::EdgeType) const;

        /// return an array of the edge measures on the quadrature rule. The output
        /// has shape (n, n_edges) where n is the length of the quadrature rule.
        const double *edge_measures(const QuadratureRule *) const;

        /// return an array of the edge measures on the quadrature rule for edges of
        /// the requested types. The output has shape (n, n_edges) where n is the
        /// length of the quadrature rule.
        const double *edge_measures(const QuadratureRule *, Edge::EdgeType) const;

    /// @brief constructs a mesh of QuadElements given a list vertices x and a
    /// list of indices indicating the vertices of each element. 
        /// @param[in] nx number of vertices
        /// @param[in] x shape (2, nx). The coordinates of the vertices
        /// @param[in] nel number of elements
        /// @param[in] elems shape (4, nel). The element corners. if j = elems(i, el)
    /// then the i-th corner of element el is x(*, j). 
        /// @return mesh
        static Mesh2D from_vertices(int nx, const double *x, int nel, const int *elems);

        /// @brief loads a mesh of QuadElements from files in dir. Must be in
        /// the mesh format as described in the docs.
        /// @param[in] dir directory with mesh files.
        /// @return mesh
        static Mesh2D from_file(const std::string& dir);

        /// @brief construct a uniform structured mesh for the rectangle [ @a ax , @a bx ] x [ @a ay , @a by ] with @a nx by @a ny elements.
        /// @param nx number of elements to partition [ @a ax , @a bx ]
        /// @param ax lower bound for x
        /// @param bx upper bound for x
        /// @param ny number of elements to partition [ @a ay , @a by ]
        /// @param ay lower bound for y
        /// @param by upper bound for y
        /// @return the mesh.
        static Mesh2D uniform_rect(int nx, double ax, double bx, int ny, double ay, double by);

    private:
        // resets all data
        inline void reset()
        {
            _edges.clear();
            _elements.clear();
            _boundary_edges.clear();
            _interior_edges.clear();
            J.clear();
            detJ.clear();
            x.clear();
            n.clear();
            edge_x.clear();
            edge_meas.clear();
            n_int.clear();
            edge_x_int.clear();
            edge_meas_int.clear();
            n_ext.clear();
            edge_x_ext.clear();
            edge_meas_ext.clear();
        }
    
        template <typename MetricEval>
        const double * element_metric(int dim, MetricCollection& map, const QuadratureRule * quad, MetricEval fun) const;

        template <bool byEdgeType, typename MetricEval>
        const double * edge_metric(int dim, Edge::EdgeType etype, MetricCollection& map, const QuadratureRule * quad, MetricEval fun) const;
    };
} // namespace dg


#endif