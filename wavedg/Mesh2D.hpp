#ifndef DG_MESH2D_HPP
#define DG_MESH2D_HPP

#include <vector>
#include <cmath>
#include <unordered_map>
#include <memory>

#include "config.hpp"
#include "QuadratureRule.hpp"
#include "Tensor.hpp"
#include "Element.hpp"
#include "Edge.hpp"

namespace dg
{
    class Mesh2D
    {
    private:
        std::vector<std::unique_ptr<Edge>> _edges;
        std::vector<std::unique_ptr<Element>> _elements;
        std::vector<int> _boundary_edges;
        std::vector<int> _interior_edges;

        mutable std::unordered_map<const QuadratureRule *, std::unique_ptr<double[]>> J;
        mutable std::unordered_map<const QuadratureRule *, std::unique_ptr<double[]>> detJ;
        mutable std::unordered_map<const QuadratureRule *, std::unique_ptr<double[]>> x;
        
        mutable std::unordered_map<const QuadratureRule *, std::unique_ptr<double[]>> n;
        mutable std::unordered_map<const QuadratureRule *, std::unique_ptr<double[]>> edge_x;
        mutable std::unordered_map<const QuadratureRule *, std::unique_ptr<double[]>> edge_meas;

        mutable std::unordered_map<const QuadratureRule *, std::unique_ptr<double[]>> n_int;
        mutable std::unordered_map<const QuadratureRule *, std::unique_ptr<double[]>> edge_x_int;
        mutable std::unordered_map<const QuadratureRule *, std::unique_ptr<double[]>> edge_meas_int;
        
        mutable std::unordered_map<const QuadratureRule *, std::unique_ptr<double[]>> n_ext;
        mutable std::unordered_map<const QuadratureRule *, std::unique_ptr<double[]>> edge_x_ext;
        mutable std::unordered_map<const QuadratureRule *, std::unique_ptr<double[]>> edge_meas_ext;

    public:
    Mesh2D() {}
    Mesh2D(const Mesh2D& mesh) = delete;
    Mesh2D(Mesh2D&&) = default;

    Mesh2D& operator=(const Mesh2D&) = delete;
    Mesh2D& operator=(Mesh2D&&) = default;

    int n_elem() const
    {
        return _elements.size();
    }

    int n_edges() const
    {
        return _edges.size();
    }

    int n_edges(EdgeType type) const
    {
        if (type == BOUNDARY)
        {
            return _boundary_edges.size();
        }
        else
        {
            return _interior_edges.size();
        }
    }

    const Edge * edge(int i) const
    {
        #ifdef WDG_DEBUG
        if (i < 0 || i >= (int)_edges.size())
            throw std::out_of_range("edge index out of range");
        #endif

        return _edges[i].get();
    }

    const Edge * edge(int i, EdgeType type) const
    {
        if (type == BOUNDARY)
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

    const Element * element(int el) const
    {
        #ifdef WDG_DEBUG
        if (el < 0 || el >= (int)_elements.size())
            throw std::out_of_range("element index out of range.");
        #endif

        return _elements[el].get();
    }

    // returns an array of the element jacobians evaluated on a quadrature rule.
    // The output J has shape (2, 2, n, n, n_elem) where n is the length of the
    // quadrature rule.
    const double * element_jacobians(const QuadratureRule *) const;

    // returns an array of the element measures ie the determinant of the
    // jacobians evaluated on a quadrature rule. The output detJ has shape (n,
    // n, n_elem) where n is the length of the quadrature rule.
    const double * element_measures(const QuadratureRule *) const;
    
    // returns an array of the physical coordinates of the quadrature rule on
    // every element. The output x has shape (2, n, n, n_elem) where n is the
    // length of the quadrature rule.
    const double * element_physical_coordinates(const QuadratureRule *) const;

    // returns an array of the normal derivatives of all of the edges evaluated
    // on the quadrature rule. The output has shape (2, n, n_edges) where n is
    // the length of the quadrature rule.
    const double * edge_normals(const QuadratureRule *) const;

    // returns an array of the normal derivatives of all of the edges of the
    // requested EdgeType evaluated on the quadrature rule. The output has shape
    // (2, n, n_edges) where n is the length of the quadrature rule.
    const double * edge_normals(const QuadratureRule *, EdgeType) const;

    // returns an array of the physical coordinates of the quadrature rule on
    // every edge. The output has shape (2, n, n_edges) where n is the length of
    // the quadrature rule.
    const double * edge_physical_coordinates(const QuadratureRule *) const;

    // returns an array of the physical coordinates of the quadrature rule on
    // every edge of the requested type. The output has shape (2, n, n_edges)
    // where n is the length of the quadrature rule.
    const double * edge_physical_coordinates(const QuadratureRule *, EdgeType) const;

    // return an array of the edge measures on the quadrature rule. The output
    // has shape (n, n_edges) where n is the length of the quadrature rule.
    const double * edge_measures(const QuadratureRule *) const;

    // return an array of the edge measures on the quadrature rule for edges of
    // the requested types. The output has shape (n, n_edges) where n is the
    // length of the quadrature rule.
    const double * edge_measures(const QuadratureRule *, EdgeType) const;

    /// @brief constructs a mesh of QuadElements given a list vertices x and a
    /// list of indices indicating the vertices of each element. 
    /// @param nx number of vertices
    /// @param x shape (2, nx). The coordinates of the vertices
    /// @param nel number of elements
    /// @param elems shape (4, nel). The element corners. if j = elems(i, el)
    /// then the i-th corner of element el is x(*, j). 
    static Mesh2D from_vertices(int nx, const double * x, int nel, const int * elems);
    };
} // namespace dg


#endif