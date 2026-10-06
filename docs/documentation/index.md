# Documentation

Build and solve quadratic and conic optimization problems with PDHCG, an
open-source first-order solver accelerated by NVIDIA GPUs.

<div class="doc-paths" markdown>

[**Install PDHCG**<br>Set up Python or build the native solver.](../installation.md)

[**Solve your first problem**<br>Model a QP or a conic problem in Python.](../python/quickstart.md)

[**Explore examples**<br>Work with low-rank objectives and cone constraints.](../examples.md)

</div>

## Choose your interface

| Interface | Use it for | Start here |
| --- | --- | --- |
| Python | Model with NumPy and SciPy; inspect solutions and warm start. | [Quick start](../python/quickstart.md) · [Model API](../python/model.md) |
| CVXPY | Use PDHCG as the solver for a CVXPY model. | [CVXPY integration](../python/quickstart.md#cvxpy) |
| C | Integrate the solver into a native application. | [C API overview](../c/overview.md) |
| Command line | Solve problem files or run across multiple GPUs. | [Build and install](../installation.md#c-executable) |

!!! info "GPU requirements"
    PDHCG requires an NVIDIA GPU with CUDA 12.4+. The Python interface uses a
    single GPU. Distributed solving uses the native executable, built with
    MPI and NCCL; see [multi-GPU setup](../installation.md#build-with-multi-gpu-support).

## Problem Formulation

PDHCG solves quadratic conic programs in the following form:

$$
\begin{aligned}
\min_{x} \quad & \frac{1}{2}x^\top (Q + R^\top D R) x + c^\top x \\
\text{s.t.} \quad & \ell_c \le Ax \le u_c, \\
                  & Fx + g \in \mathcal{K}_a, \\
                  & \ell_v \le x \le u_v, \\
                  & x_J \in \mathcal{K}_v \quad \text{for variable-cone blocks } J.
\end{aligned}
$$

Where:

- $Q$ is a sparse symmetric matrix (optional)
- $R \in \mathbb{R}^{k\times n}$ is a low-rank factor of rank $k$ (optional)
- $D \in \mathbb{R}^{k\times k}$ is an optional middle matrix; defaults to the identity, recovering the standard $Q + R^\top R$ form. May be diagonal, sparse, dense, or indefinite — the backend auto-detects the cheapest representation
- $A$ is the constraint matrix
- $F$ and $g$ define the native affine-cone map
- $c$ is the linear objective vector
- $\ell_c, u_c$ are constraint bounds
- $\ell_v, u_v$ are variable bounds
- $\mathcal{K}_a$ and $\mathcal{K}_v$ are products of Standard SOC, Rotated SOC, Exponential, Power, or positive-semidefinite cones

## Next steps

- [Solver parameters](../python/parameters.md): tolerances, termination, scaling, and restart behavior.
- [C types](../c/types.md) and [functions](../c/functions.md): the native solver interface.
- [Migration guide](../migration.md): update code to the 0.3 API.
- [Algorithm](../algorithm.md): standard form, primal–dual formulation, and PDHG updates.
- [Citation](../citation.md): the latest PDHCG paper and BibTeX.
