# waveDG
This library implements the Discontinuous Galerkin Finite Element Method in 2D with emphasis on application to the linear wave equation as a first order system (e.g. linearized Euler equations):
$$p_t + \nabla \cdot \mathbf{u} = 0,$$
$$\mathbf{u}_t + \nabla p = 0.$$

The code currently supports Dirichlet, Neumann, and (approximate) outflow boundary conditions.

## DG Formulation
For the linear hyperbolic conservation law:
$$u_t + \nabla\cdot (\mathbf{A} u) = u_t + \partial_x (A^0 u) + \partial_y (A^1 u) = 0.$$
Let \f$I\f$ be an element on the mesh $\Omega_h$. We define the approximation space \f$V_h\subset L^2(I)\f$ to be the space of polynomials upto degree \f$P\f$ on \f$I\f$. The DG weak form of the equation on each element \f$I\f$ is
$$(u_t, v)_I - (A^0 u, v_x)_I - (A^1 u, v_y)_I + \langle u^\star, [v] \rangle\_{\partial I} = 0, \qquad \forall v\in V_h.$$
Where
$$u^\star = a C\\{u\\} + b |C|\mathbf\[u\], \\quad C = \mathbf{n}\cdot\mathbf{A} = n_x A^0 + n_y A^1.$$
Here \f$\mathbf{n}\f$ is the normal to \f$\partial I\f$, and if \f$C = R\Lambda R^{-1}\f$ is the eigenvalue decomposition of \f$C\f$, then \f$|C| = R|\Lambda|R^{-1}\f$ where \f$|\Lambda|\f$ has the absolute values of the eigenvalues of \f$C\f$ on the diagonal.
Note that it is necessary that \f$C\f$ have real eigenvalues for any \f$\mathbf{n}\f$, which is also a condition for the well-posedness of the PDE.
Note that the upwind scheme corresponds to \f$a = 1, b = -1/2\f$, and the central (average) scheme corresponds to \f$a = 1, b = 0\f$.

## Conventions

> The "shape" of the data is frequently identified throughout the code and documentation. The shape refers to the layout of high dimension (tensor) data in memory.
> The shape `(n,)` means an array of length n; the shape `(m, n)` means a matrix with m rows and n columns and is ordered in column major format.
> For higher dimensional tensors `(n1, n2, ...)` the memory is ordered recursively in column major, meaning the first dimension is continuous, the second dimension is spaced by blocks of size `n1`, the third by blocks of size `n1*n2`, etc. To abstract away the memory layout, `TensorWrapper` and `Tensor` classes are used extensively throughout the code. The `TensorWrapper` "wraps" an externally managed array and provides high dimensional indexing for that array, and the `Tensor` class is a `TensorWrapper` that manages its own memory.

Given a mesh \f$\Omega_h\f$ with \f$n\f$ elements, and a quadrature rule \f$Q\f$ with \f$p\f$ collocation points, a function on the mesh \f$u : \Omega_h\to\mathbb{R}^d\f$ to a PDE is represented by its nodal values on each element on of the mesh.
On each element, we assume a tensor product basis. The basis functions are the Lagrange interpolating polynomials on the collocation points of \f$Q\f$.
The convention is to organize the degrees of freedom by locality, so \f$u\f$ has shape `(d, p, p, n)`. Therefore on each collocation point, the `d` dimensions of \f$u\f$ are continuous. The layout of each element is determined from the reference coordinates \f$(\xi,\eta)\f$ with the \f$\xi\f$ direction ordered first.
> Sometimes when \f$d=1\f$, then the first dimension is omitted from the shape. That is, \f$u\f$ has shape `(p, p, n)`.

> Throughout the code `n_var` refers to the vector dimension of \f$u\f$, that is `n_var` \f$=d\f$.