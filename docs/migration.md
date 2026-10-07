# Migration Guide

## Migrating to 0.4

- CPU and CUDA have independent implementations behind shared device interfaces.
  All interfaces use the same native device selection; see [Devices](devices.md).
- `eps_infeas_detect` has been removed. Use `eps_infeasible` in C and the CLI,
  or [`InfeasibleTol`](python/parameters.md#termination-criteria) in Python.
- Recompile C clients against the updated structures, and initialize parameters
  with `set_default_parameters`.

## Migrating to 0.3

Version 0.3 adds quadratic conic models and changes the C and Python model
construction APIs. This page covers the source changes needed by existing 0.2
callers.

### C API

[`create_qp_problem`](c/functions.md#create_qp_problem) has six new trailing
arguments for variable and affine cones. Append neutral values for a plain QP:

```c
qp_problem_t *problem = create_qp_problem(
    c, Q, R, D, A, con_lb, con_ub, var_lb, var_ub, objective_constant,
    0, NULL, NULL, NULL, 0, NULL);
```

For conic inputs and coordinate conventions, see
[`cone_spec_t`](c/types.md#cone-spec).

`pdhcg_postsolve` now returns nonzero after a complete primal-dual recovery and
zero when postsolve fails or full dual recovery is unavailable:

```c
if (!pdhcg_postsolve(info, result, original_problem)) {
    /* Handle postsolve failure. */
}
```

`qp_problem_t` and `pdhg_parameters_t` gained conic fields. Do not depend on
their old binary layout. Recompile downstream code and initialize parameters
through `set_default_parameters` before overriding individual fields.

### Python API

Cone metadata is now columnar. Replace a list of dictionaries with one
`ConeSpec`:

```python
import numpy as np
from pdhcg import ConeSpec, ConeType

cones = ConeSpec(
    types=np.array([ConeType.SOC, ConeType.POWER], dtype=np.int32),
    starts=np.array([0, 4], dtype=np.int32),
    v_dims=np.array([2, 1], dtype=np.int32),
    power_alphas=np.array([0.0, 0.4]),
)
```

Pass this object as `variable_cones` or `affine_cones` when constructing a
`Model`. Legacy `list[dict]` inputs intentionally raise `TypeError`.

CVXPY is an optional integration; see the
[CVXPY quick start](python/quickstart.md#cvxpy).

### Executable Location

A source build places the command-line executable at `build/pdhcg`. Installed
packages place it in the installation prefix's `bin` directory.
