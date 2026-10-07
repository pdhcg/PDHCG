# PDHCG: A First-Order Solver for Quadratic Conic Programming with (Multi-) GPU Acceleration

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE) [![PyPI version](https://badge.fury.io/py/pdhcg.svg)](https://pypi.org/project/pdhcg/) [![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-blue.svg)](https://lhongpei.github.io/PDHCG) [![Publication](https://img.shields.io/badge/DOI-10.1287/ijoc.2024.0983-B31B1B.svg)](https://pubsonline.informs.org/doi/10.1287/ijoc.2024.0983) [![arXiv](https://img.shields.io/badge/arXiv-2608.09159-b31b1b.svg)](https://arxiv.org/abs/2608.09159) [![qpsolvers](https://img.shields.io/badge/qpsolvers-supported-brightgreen.svg)](https://github.com/qpsolvers/qpsolvers) [![CVXPY](https://img.shields.io/badge/CVXPY-supported-brightgreen.svg)](https://www.cvxpy.org/)

**PDHCG** is an open-source first-order solver based on primal-dual hybrid
gradient methods for large-scale convex quadratic and quadratic conic programming.
It supports CPU and CUDA backends, including distributed multi-GPU execution.

Supported models include linear programs, sparse and structured low-rank quadratic
objectives, and SOC, rotated SOC, exponential, power, and PSD cones.

PDHCG also integrates with [CVXPY](docs/python/quickstart.md#cvxpy) and
[qpsolvers](https://github.com/qpsolvers/qpsolvers).

## Standard Form

$$
\begin{aligned}
\min_x\quad &\tfrac12x^\top(Q+R^\top D R)x+c^\top x\\
\text{s.t.}\quad &\ell_c\le Ax\le u_c,\\
&Fx+g\in\mathcal K_a,\\
&\ell_v\le x\le u_v,\\
&x_J\in\mathcal K_v.
\end{aligned}
$$

The full Hessian $H=Q+R^\top D R$ is positive semidefinite.
See [Algorithm](docs/algorithm.md#standard-form) for notation and solver details.

## Documentation

| Topic | Reference |
| --- | --- |
| Requirements and installation | [Installation](docs/installation.md) |
| Backends, threads, and multi-GPU execution | [Devices](docs/devices.md) |
| Python getting started | [Quick start](docs/python/quickstart.md) |
| Python API | [Model](docs/python/model.md), [Parameters](docs/python/parameters.md) |
| Native API | [C API](docs/c/overview.md) |
| Command-line options | [CLI](docs/cli.md) |
| Modeling examples | [Examples](docs/examples.md) |
| Upgrading | [Migration guide](docs/migration.md) |

## Citation

If you use PDHCG in research, see the [citation and BibTeX](docs/citation.md).

## Acknowledgments

This solver is built upon the infrastructure of [cuPDLPx](https://github.com/MIT-Lu-Lab/cuPDLPx) (originally developed by Haihao Lu). We gratefully acknowledge this project for providing the high-performance CUDA-C framework for Linear Programming (LP) that serves as the foundation for this QP solver.



---

## License

Copyright 2024-2026 Hongpei Li, Haihao Lu.

Licensed under the Apache License, Version 2.0. See the [LICENSE](LICENSE) file for details.
