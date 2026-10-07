# Python Quick Start

After [installing PDHCG](../installation.md), define a model with NumPy or
SciPy arrays and call `optimize()`. See [devices](../devices.md) for backend
selection and execution settings.

## Basic QP

Solve $\min x_1^2 + x_2^2 - 2x_1 - 6x_2$ subject to
$x_1 + x_2 \le 2$ and $x \ge 0$:

```python
import numpy as np
import scipy.sparse as sp
from pdhcg import Model

model = Model(
    objective_matrix=sp.diags([2.0, 2.0]),
    objective_vector=np.array([-2.0, -6.0]),
    constraint_matrix=sp.csr_matrix([[1.0, 1.0]]),
    constraint_upper_bound=np.array([2.0]),
    variable_lower_bound=np.zeros(2),
)
model.optimize()
print(model.Status, model.ObjVal, model.X)
```

The optimum is $x=(0,2)$ with objective $-8$.

## Quick Start with Cone Constraints

Minimize $z$ with $(v,w,z)$ in a standard second-order cone and
$v=3$, $w=4$. The optimum is $(3,4,5)$.

```python
import numpy as np
import scipy.sparse as sp
from pdhcg import ConeSpec, ConeType, Model

model = Model(
    objective_vector=np.array([0.0, 0.0, 1.0]),
    constraint_matrix=sp.csr_matrix([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
    constraint_lower_bound=np.array([3.0, 4.0]),
    constraint_upper_bound=np.array([3.0, 4.0]),
    variable_cones=ConeSpec(ConeType.SOC, np.array([0], dtype=np.int32)),
)
model.optimize()
print(model.Status, model.X)
```

See the [model reference](model.md) for constructor arguments,
[ConeSpec](model.md#cone-constraints), [warm starts](model.md#warm-starts),
and [results](model.md#results). Solver settings belong in the
[parameter reference](parameters.md).

## CVXPY

With the [CVXPY extra installed](../installation.md#cvxpy), import the backend
once to register `solver="PDHCG"`:

```python
import cvxpy as cp
import pdhcg.cvxpy_backend  # Registers solver="PDHCG".

x = cp.Variable()
problem = cp.Problem(cp.Minimize(x), [x >= 1])
value = problem.solve(solver="PDHCG", eps=1e-6)
print(problem.status, value, x.value)
```

The integration supports quadratic objectives and Zero, NonNeg, SOC, PSD,
ExpCone, and PowCone3D constraints, with CVXPY's primal and dual conventions.
Mixed-integer models are not supported.
