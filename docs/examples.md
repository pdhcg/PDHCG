# Examples

Start with the [Python quick start](python/quickstart.md) for a basic QP or SOC
model. These examples cover structured objectives and cone modeling; see
[devices](devices.md) for backend selection, threads, and multi-GPU execution.

## Low-Rank Quadratic Term

Keep $R^\top D R$ factored instead of forming the full Hessian. Here
$Q=I$ and $D$ has positive diagonal entries, so the objective is strictly convex.

```python
import numpy as np
import scipy.sparse as sp
from pdhcg import Model

Q = sp.eye(4, format="csr")
R = np.array([[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, -1.0]])
D = np.array([0.5, 2.0])
c = np.array([-1.0, -2.0, -3.0, -4.0])

model = Model(
    objective_matrix=Q,
    objective_matrix_low_rank=R,
    objective_matrix_low_rank_middle=D,
    objective_vector=c,
)
model.optimize()
print(model.Status, model.X)  # Approximately [0.25, 1.25, 3.4, 3.6].
```

Omit `objective_matrix_low_rank_middle` to use $D=I$. The
[model reference](python/model.md#construction-and-updates) describes the
other supported representations of `D`.

## Warm Starting

Cache the previous solution before updating the model, then reuse it for the
modified objective:

```python
import numpy as np
import scipy.sparse as sp
from pdhcg import Model

model = Model(
    objective_matrix=sp.diags([2.0, 2.0]),
    objective_vector=np.array([-2.0, -6.0]),
)
model.optimize()
x0, y0 = model.X.copy(), model.Pi.copy()

model.setObjectiveVector(np.array([-2.5, -6.5]))
model.setWarmStart(primal=x0, dual=y0)
model.optimize()
print(model.X)  # Approximately [1.25, 3.25].
```

## Sparse Quadratic Objective Coupled to SOC

Variables are $(a,b,v,w,z)$, with $a=v$, $b=w$, and $(v,w,z)$ in a standard
second-order cone. The non-diagonal quadratic term acts on $(a,b)$.
The optimum is $(3,4,3,4,5)$, with objective $-18.5$.

```python
import numpy as np
import scipy.sparse as sp
from pdhcg import ConeSpec, ConeType
from pdhcg._core import solve_once

A = sp.csr_matrix([[1.0, 0.0, -1.0, 0.0, 0.0],
                   [0.0, 1.0, 0.0, -1.0, 0.0]])
Q = sp.csr_matrix([[1.0, 0.5, 0.0, 0.0, 0.0],
                   [0.5, 1.0, 0.0, 0.0, 0.0],
                   [0.0, 0.0, 0.0, 0.0, 0.0],
                   [0.0, 0.0, 0.0, 0.0, 0.0],
                   [0.0, 0.0, 0.0, 0.0, 0.0]])
c = np.array([-5.6, -6.3, 0.0, 0.0, 1.0])

info = solve_once(
    Q=Q, R=None, A=A, objective_vector=c,
    constraint_lower_bound=np.zeros(2),
    constraint_upper_bound=np.zeros(2),
    cones=ConeSpec(ConeType.SOC, np.array([2], dtype=np.int32)),
)
print(info["Status"], info["X"], info["PrimalObj"])
```

## Exponential Cone with Fixed Coordinates

A quasi-linear Fisher market with two buyers and three goods uses allocations
$x_{ij}$, slack $v_i$, and one exponential-cone triple $(z_i,y_i,t_i)$ per
buyer. Fixing $y_i=1$ expresses $\exp(z_i)\le t_i$.

```python
import numpy as np
import scipy.sparse as sp
from pdhcg import ConeSpec, ConeType
from pdhcg._core import solve_once

u = np.array([[0.5, 1.0, 0.2], [0.3, 0.7, 1.0]])  # Utilities.
w = np.array([1.0, 1.5])                           # Budgets.
b = np.array([1.0, 1.0, 1.0])                      # Supplies.
n, m = u.shape
nx, N = n*m, n*m + n + 3*n                         # x | v | (z,y,t).
v0, c0 = nx, nx + n

# Supply rows followed by utility-balance rows.
rows = sp.lil_matrix((m + n, N))
for j in range(m):
    rows[j, [i*m + j for i in range(n)]] = 1.0      # sum_i x_ij = b_j.
for i in range(n):
    rows[m+i, i*m:(i+1)*m] = -u[i]                 # -u_i^T x_i - v_i + t_i = 0.
    rows[m+i, v0+i] = -1.0
    rows[m+i, c0+3*i+2] = 1.0
A = rows.tocsr()

c = np.zeros(N)
for i in range(n):
    c[v0+i] = 1.0
    c[c0+3*i] = -w[i]                              # min sum_i v_i - w_i z_i.
lb = np.full(N, -np.inf)
lb[:nx] = 0.0
lb[v0:v0+n] = 0.0
con_b = np.concatenate([b, np.zeros(n)])

primal_start = np.zeros(N)
primal_start[c0+1::3] = 1.0                        # Values for fixed y_i.
fixed_mask = np.zeros(N, dtype=np.uint8)
fixed_mask[c0+1::3] = 1
cones = ConeSpec(
    ConeType.EXP,
    c0 + 3 * np.arange(n, dtype=np.int32),
    fixed_mask=fixed_mask,
)

info = solve_once(
    Q=None, R=None, A=A, objective_vector=c,
    variable_lower_bound=lb,
    constraint_lower_bound=con_b,
    constraint_upper_bound=con_b,
    primal_start=primal_start,
    cones=cones,
)
x = info["X"]
for i in range(n):
    z, y, t = x[c0+3*i:c0+3*i+3]
    print(f"buyer {i}: y={y:.4f}  exp(z/y)={np.exp(z/y):.4f}  t={t:.4f}")
```
