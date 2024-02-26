# waveDG
This library implements the Discontinuous Galerkin Finite Element Method in 2D with emphasis on application to the linear wave equation as a first order system (e.g. linearized Euler equations):
$$p_t + \nabla \cdot \mathbf{u} = 0,$$
$$\mathbf{u}_t + \nabla p = 0.$$

The code currently supports Dirichlet, Neumann, and (approximate) outflow boundary conditions.

## DG Formulation
For the linear hyperbolic conservation law:
$$\mathbf{q}_t + \partial_x (A \mathbf{q}) + \partial_y (B \mathbf{q}) = 0.$$
Let $I$ be an element on the mesh. We define the approximation space $V_h\subset L^2(I)$ to be the space of polynomials upto degree $P$ on $I$. The DG weak form of the equation on each element $I$ is
$$(\mathbf{q}_t, v)_I - (A\mathbf{q}, v_x)_I - (B \mathbf{q}, v_y)_I + \langle \mathbf{q}^\star, \[v\] \rangle\_\{\partial I\} = 0, \\qquad \forall v\in V_h.$$
Where
$$\mathbf{q}^\star = a C\\{\mathbf{q}\\} + b |C|\mathbf\[\mathbf{q}\], \\quad C = n_x A + n_y B.$$
Here $(n_x, n_y)$ is the normal to $\partial I$, and if $C = R\Lambda R^{-1}$ is the eigenvalue decomposition of $C$, then $|C| = R|\Lambda|R^{-1}$ where $|\Lambda|$ has the absolute values of the eigenvalues of $C$ on the diagonal.
Note that it is necessary that $C$ have real eigenvalues for any $(n_x, n_y)$, which is also a condition for the well-posedness of the PDE.
We can choose $a = 1, b = 1/2$ for the upwind scheme, or $a = 1, b = 0$ for the central (average) scheme.

## Compiling the Library
To compile the serial version of the library run:
```
cmake .
make wavedg -j
```
To compile the MPI version of the library run:
```
cmake . -D WDG_USE_MPI=ON
make wavedg -j
```
In either cases, you can add the flag `-D WDG_DEBUG=ON` to enable debugging (`-g`) and add additional correctness checks.

To install the library run `sudo make install`.