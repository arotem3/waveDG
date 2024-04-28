#ifndef DG_MESH2D_HPP
#define DG_MESH2D_HPP

#include <vector>
#include <cmath>
#include <unordered_map>
#include <memory>
#include <queue>
#include <unordered_set>
#include <set>
#include <fstream>
#include <algorithm>

#include "wdg_config.hpp"
#include "QuadratureRule.hpp"
#include "Tensor.hpp"
#include "Element.hpp"
#include "Edge.hpp"
#include "Node.hpp"

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
    public:
        class ElementMetricCollection
        {
        public:
            /// @brief initialize collection of metrics on elements
            /// @param mesh the mesh
            /// @param quad the quadrature rule to evaluate metrics on
            ElementMetricCollection(const Mesh2D& mesh, const QuadratureRule * quad);

            ElementMetricCollection(ElementMetricCollection&&);

            /// returns an array of the element jacobians evaluated on a quadrature rule.
            /// The output J has shape (2, 2, n, n, n_elem) where n is the length of the
            /// quadrature rule.
            const double * jacobians() const;
            
            /// returns an array of the element measures ie the determinant of the
            /// jacobians evaluated on a quadrature rule. The output detJ has shape (n, n, n_elem)
            /// where n is the length of the quadrature rule.
            const double * measures() const;
            
            /// returns an array of the physical coordinates of the quadrature rule on
            /// every element. The output x has shape (2, n, n, n_elem) where n is the
            /// length of the quadrature rule.
            const double * physical_coordinates() const;

        private:
            const Mesh2D& mesh;
            const QuadratureRule * const quad;

            mutable std::unique_ptr<double[]> J;
            mutable std::unique_ptr<double[]> detJ;
            mutable std::unique_ptr<double[]> x;
        };

        class EdgeMetricCollection
        {
        public:
            /// @brief initialize collection of metrics on edges
            /// @param mesh the mesh
            /// @param quad the quadrature rule to evaluate metrics on
            /// @param edge_type the type of edges to evaluate metrics for (interior or boundary)
            EdgeMetricCollection(const Mesh2D& mesh, const QuadratureRule * quad, const FaceType edge_type);

            EdgeMetricCollection(EdgeMetricCollection&&);

            /// return an array of the edge measures on the quadrature rule for edges of
            /// the requested types. The output has shape (n, n_edges) where n is the
            /// length of the quadrature rule.
            const double * measures() const;

            /// returns an array of the physical coordinates of the quadrature rule on
            /// every edge of the requested type. The output has shape (2, n, n_edges)
            /// where n is the length of the quadrature rule.
            const double * physical_coordinates() const;

            /// returns an array of the normal derivatives of all of the edges of the
            /// requested FaceType evaluated on the quadrature rule. The output has shape
            /// (2, n, n_edges) where n is the length of the quadrature rule.
            const double * normals() const;

        private:
            const Mesh2D& mesh;
            const QuadratureRule * const quad;
            const FaceType edge_type;

            mutable std::unique_ptr<double[]> detJ;
            mutable std::unique_ptr<double[]> x;
            mutable std::unique_ptr<double[]> n;
        };

        /// @brief constructs empty mesh
        Mesh2D() {}
        ~Mesh2D() = default;

        /// @brief DELETED: mesh maintains unique pointers to abstract types. Copies are non-trivial.
        Mesh2D(const Mesh2D &mesh) = delete;

        /// @brief move mesh
        Mesh2D(Mesh2D &&) = default;

        /// @brief DELETED: mesh maintains unique pointers to abstract types. Copies are non-trivial. 
        Mesh2D &operator=(const Mesh2D &) = delete;

        /// @brief move mesh 
        Mesh2D &operator=(Mesh2D &&) = default;

    #ifdef WDG_USE_MPI
        /// @brief Distributes mesh from root to all other processors on comm.
        /// Attempts to find a load balanced distribution which minimizes shared
        /// edges.
        ///
        /// @details
        /// **recursive coordinate bisecion algorithm:** @n
        /// (1) set P = # of processors. @n
        /// (2) if P == 1: @n
        ///         return. @n
        ///     else: @n
        ///     (3) compute the geometric center of each element. @n
        ///     (4) compute the first principal component of the element centers to
        /// determine the direction of greatest variance. @n
        ///     (5) set n = smallest prime divisor of P. @n
        ///     (6) set y = inner product of element center and first principal component. @n
        ///     (7) split elements into n equal groups sorted by y. @n
        ///     (8) For each of the n groups, repeat (2-8) with P = P/n. @n
        /// 
        /// This strategy ensures that elements that are geometrically close
        /// will be grouped together. The split is computed recursively into the
        /// smallest # of groups possible at each step rather than in one step
        /// to ensure that groups of elements are tightly clustered rather than
        /// spread out. e.g. given a uniform mesh on [0,1]x[0,1] with 4
        /// processors, if we do not split the mesh recursively but instead, at
        /// step (7) split the mesh into 4 pieces, we may end up with the split
        /// [0, 1/4]x[0, 1], [1/4, 1/2]x[0, 1], etc i.e. long narrow groups of
        /// elements, hence high # of shared edges between processors. If,
        /// instead we proceed with the recursion, then split should be the
        /// optimal [0, 1/2]x[0, 1/2], [1/2, 1]x[0, 1/2] etc. @n
        ///
        /// **reverse Cuthill-McKee algorithm:** @n
        /// Constructs adjacency matrix for the connectivity of elements along
        /// edges, and performs symmetric RCM reordering of the rows and columns
        /// of the matrix to minimize the bandwidth of the adjacency matrix.
        /// Minimizing the bandwidth effectively orders the elements so that
        /// elements are ordered near their neighbors. With this ordering, we
        /// simply take the partition in order, e.g. first N/2 elements go to
        /// processor 1, second N/2 elements got to processor 2. See
        /// https://en.wikipedia.org/wiki/Cuthill%E2%80%93McKee_algorithm
        ///
        /// @param[in] comm the communicator over which to distribute mesh
        /// @param[in] alg either "rcb" (recursive coordinate bisection) or "rcm"
        /// (reverse Cuthill-McKee).
        void distribute(const std::string& alg = "rcm");

        /// total number of elements in the distributed mesh. This function is
        /// blocking and must be called from all processors.
        int global_n_elem() const;

        /// returns the rank of the processors owning element `el`.
        int find_element(int el) const
        {
        #ifdef WDG_DEBUG
            if (el < 0 || el >= (int)e2p.size())
                wdg_error("Mesh2D::find_element error: element index out of range.");
        #endif
            return e2p[el];
        }

        /// @brief returns the local index of an edge given its global index, e.g. by Edge->id; Edge must be on processor. 
        int local_edge_index(int global_edge_index) const
        {
        #ifdef WDG_DEBUG
            if (not _edge_local_id.contains(global_edge_index))
                wdg_error("Mesh2D::local_edge_index error: Edge does not belong to this processor.");
        #endif    
            return _edge_local_id.at(global_edge_index);
        }

        /// @brief returns the local index of an element given its global index, e.g. by Element->id; Element must be on processor. 
        int local_element_index(int global_element_index) const
        {
        #ifdef WDG_DEBUG
            if (not _elem_local_id.contains(global_element_index))
                wdg_error("Mesh2D::local_element_index error: Element does not belong to this processor.");
        #endif
            return _elem_local_id.at(global_element_index);
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

        /// number of elements in mesh. If mesh is distributed (MPI), then
        /// returns the number of elements on this processor.
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
        int n_edges(FaceType type) const
        {
            if (type == FaceType::BOUNDARY)
            {
                return _boundary_edges.size();
            }
            else
            {
                return _interior_edges.size();
            }
        }

        int n_nodes() const
        {
            return _nodes.size();
        }

        int n_nodes(NodeType type) const
        {
            if (type == NodeType::INTERIOR)
                return _interior_nodes.size();
            else
                return _boundary_nodes.size();
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

        /// @brief returns the shortest length scale (edge length) of the mesh.
        double min_h() const;

        /// @brief returns the longest length scale (edge length) of the mesh.
        double max_h() const;

        const Node& node(int i) const
        {
        #ifdef WDG_DEBUG
            if (i < 0 || i >= (int)_nodes.size())
                wdg_error("Mesh2D::node error: node index out of range.");
        #endif

            return _nodes[i];
        }

        const Node& node(int i, NodeType type) const
        {
            if (type == NodeType::BOUNDARY)
            {
            #ifdef WDG_DEBUG
                if (i < 0 || i >= (int)_boundary_nodes.size())
                    wdg_error("Mesh2D::node error: boundary node index out of range.");
            #endif

                return _nodes[_boundary_nodes[i]];
            }
            else
            {
            #ifdef WDG_DEBUG
                if (i < 0 || i >= (int)_interior_nodes.size())
                    wdg_error("Mesh2D::node error: interior node index out of range.");
            #endif
            
                return _nodes[_interior_nodes[i]];
            }
        }

        /// returns the edge specified by edge index i. For distributed meshes:
        /// this index is local to the processor and should be in the range [0,
        /// n_edges() ).
        const Edge *edge(int i) const
        {
        #ifdef WDG_DEBUG
            if (i < 0 || i >= (int)_edges.size())
            wdg_error("Mesh2D::edge error: edge index out of range.");
        #endif

            return _edges[i].get();
        }

        /// returns the edge of FaceType type specified by edge index i. For
        /// distributed meshes: this index is local to the processor and should
        /// be in the range [0, n_edges(type) ).
        const Edge *edge(int i, FaceType type) const
        {
            if (type == FaceType::BOUNDARY)
            {
            #ifdef WDG_DEBUG
                if (i < 0 || i >= (int)_boundary_edges.size())
                    wdg_error("Mesh2D::edge error: boundary edge index out of range.");
            #endif

                return _edges[_boundary_edges[i]].get();
            }
            else
            {
            #ifdef WDG_DEBUG
                if (i < 0 || i >= (int)_interior_edges.size())
                    wdg_error("Mesh2D::edge error: interior edge index out of range.");
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
                wdg_error("Mesh2D::element error: element index out of range.");
        #endif

            return _elements[el].get();
        }

        const ElementMetricCollection& element_metrics(const QuadratureRule * quad) const;

        const EdgeMetricCollection& edge_metrics(const QuadratureRule * quad, FaceType edge_type) const;

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
            elem_collections.clear();
            interior_edge_collections.clear();
            boundary_edge_collections.clear();
        }

    #ifdef WDG_USE_MPI
        void compute_face_pattern() const;
    #endif

        private:
        std::vector<Node> _nodes;
        std::vector<std::unique_ptr<Edge>> _edges;
        std::vector<std::unique_ptr<Element>> _elements;
        
        std::vector<int> _interior_nodes;
        std::vector<int> _boundary_nodes;
        
        std::vector<int> _boundary_edges;
        std::vector<int> _interior_edges;

        mutable std::unordered_map<const QuadratureRule *, ElementMetricCollection> elem_collections;
        mutable std::unordered_map<const QuadratureRule *, EdgeMetricCollection> interior_edge_collections;
        mutable std::unordered_map<const QuadratureRule *, EdgeMetricCollection> boundary_edge_collections;

#ifdef WDG_USE_MPI
        std::vector<int> e2p;
        std::unordered_map<int, int> _edge_local_id;
        std::unordered_map<int, int> _elem_local_id;

        mutable ivec interior_face_pattern; // local face pattern
        mutable ivec boundary_face_pattern;
#endif
    };

    inline Mesh2D::ElementMetricCollection::ElementMetricCollection(const Mesh2D& mesh_, const QuadratureRule * quad_) : mesh(mesh_), quad(quad_) {}

    inline Mesh2D::ElementMetricCollection::ElementMetricCollection(ElementMetricCollection&& a) : mesh(a.mesh), quad(a.quad), J(std::move(a.J)), detJ(std::move(a.detJ)), x(std::move(a.x)) {}

    inline Mesh2D::EdgeMetricCollection::EdgeMetricCollection(const Mesh2D& mesh_, const QuadratureRule * quad_, const FaceType edge_type_) : mesh(mesh_), quad(quad_), edge_type(edge_type_) {}

    inline Mesh2D::EdgeMetricCollection::EdgeMetricCollection(EdgeMetricCollection&& a) : mesh(a.mesh), quad(a.quad), edge_type(a.edge_type), detJ(std::move(a.detJ)), x(std::move(a.x)), n(std::move(a.n)) {}
} // namespace dg

#endif