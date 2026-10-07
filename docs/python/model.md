# Model Class

`Model` accepts NumPy arrays and SciPy sparse matrices. See the
[quick start](quickstart.md) for a complete solve, [parameters](parameters.md)
for solver settings, and [devices](../devices.md) for execution options.

## Construction and Updates

The quadratic objective is $\frac12 x^\top(Q+R^\top D R)x+c^\top x+c_0$.
`D` defaults to identity and accepts a diagonal vector, dense symmetric matrix,
or SciPy sparse matrix. `D` may be indefinite provided the full Hessian
$Q+R^\top D R$ is positive semidefinite.

Omitted variable bounds are $-\infty$ and $+\infty$.
`Model.read_file(path)` reads MPS, QPS, or CBF files, including `.gz` files.

::: pdhcg.model.Model
    options:
      show_root_heading: false
      show_source: false
      show_docstring_description: false
      show_bases: false
      merge_init_into_class: false
      docstring_section_style: table
      members:
        - __init__
        - read_file
        - setObjectiveVector
        - setObjectiveConstant
        - setObjectiveMatrix
        - setObjectiveMatrixLowRank
        - setObjectiveMatrixLowRankMiddle
        - setConstraintMatrix
        - setConstraintLowerBound
        - setConstraintUpperBound
        - setVariableCones
        - setAffineConeConstraints
        - setVariableLowerBound
        - setVariableUpperBound
        - optimize

`m.ModelSense` defaults to `PDHCG.MINIMIZE`; set it to `PDHCG.MAXIMIZE`
for a maximization model. The constants are available through
`from pdhcg import PDHCG`.

## Cone Constraints

Use `ConeSpec` for variable or affine cones. It stores metadata in contiguous
NumPy arrays and broadcasts scalar fields across blocks.

| Field | Type | Meaning |
|---|---|---|
| `types` | scalar or `int32[K]` | `ConeType.SOC`, `RSOC`, `EXP`, `POWER`, or `PSD` |
| `starts` | `int32[K]` | First variable index or affine row of each block |
| `v_dims` | scalar or `int32[K]` | Length of `v` for SOC/RSOC, matrix order for PSD; default 1 |
| `power_alphas` | scalar or `float64[K]` | Power-cone exponents in `(0, 1)` |
| `fixed_mask` | optional `uint8[N]` | Ambient-coordinate mask for fixed non-PSD variable slots |

Block coordinates follow the [native cone layouts](../c/types.md#cone-type).
SOC/RSOC blocks have `v_dims + 2` entries; EXP/POWER blocks have three entries
and require `v_dims=1`. A PSD block of order `p` has `p*(p+1)//2` entries:
its `svec` array stores the lower triangle by columns, scaling off-diagonals
by `sqrt(2)`. For example, a symmetric $2\times2$ matrix uses
`[X[0, 0], sqrt(2)*X[1, 0], X[1, 1]]`.

Fixed values come from the corresponding primal warm-start entries.
PSD blocks do not support `fixed_mask`; use linear equalities to fix matrix
entries. Affine cones do not support fixed slots.

```python
import numpy as np
from pdhcg import ConeSpec, ConeType

# One SOC block occupying four entries, followed by an exponential block.
cones = ConeSpec(
    types=[ConeType.SOC, ConeType.EXP],
    starts=np.array([0, 4], dtype=np.int32),
    v_dims=[2, 1],
)
```

Pass this object as `Model(..., variable_cones=cones)`. For many identical
blocks, scalar metadata avoids a Python loop:

```python
num_cones = 1_000_000
cones = ConeSpec(ConeType.EXP, 3 * np.arange(num_cones, dtype=np.int32))
```

For affine constraints $Fx+g\in\mathcal K_a$, pass `affine_cone_matrix=F`,
`affine_cone_offset=g`, and `affine_cones=cones` to `Model`, or call
`setAffineConeConstraints(F, g, cones)`. The offset defaults to zero;
`starts` indexes rows of `F`, which the blocks must cover exactly once.

Cone arguments accept `ConeSpec`, not the former per-cone `list[dict]` format.
`ConeSpec.from_columnar(payload)` reconstructs metadata returned by the
low-level file reader.

## Warm Starts

```python
m.setWarmStart(primal=x0, dual=y0)
m.optimize()
m.clearWarmStart()
```

The primal vector has one entry per variable; the dual vector is ordered
`[dual_A, dual_F]`. Either argument can be omitted; `None` clears that
component. Incorrect dimensions raise a warning and leave the existing
component unchanged. Save a previous solution before changing coefficients,
as shown in the [reoptimization example](../examples.md#warm-starting).

## Results

After `optimize()`, these read-only attributes describe the returned iterate.
Check `Status` before treating it as an optimal solution. Changing model
coefficients or bounds clears the cached results.

| Attribute | Meaning |
|---|---|
| `X` | Primal vector |
| `Pi` | Dual vector, ordered `[dual_A, dual_F]` |
| `ObjVal`, `DualObj` | Primal and dual objectives, adjusted for `ModelSense` |
| `Gap`, `RelGap` | Absolute and relative objective gaps |
| `Status`, `StatusCode` | Termination status and numeric code |
| `IterCount` | Iteration count |
| `Runtime`, `RescalingTime` | Solve and rescaling times in seconds |
| `RelPrimalResidual`, `RelDualResidual` | Relative primal and dual residuals |
| `MaxPrimalRayInfeas`, `MaxDualRayInfeas` | Primal- and dual-ray infeasibility measures |
| `PrimalRayLinObj`, `DualRayObj` | Primal-ray linear objective and dual-ray certificate objective |

`PrimalInfeas` and `DualInfeas` alias the relative residuals. Result attributes
are `None` before solving or after they have been cleared.

| `Status` | `StatusCode` |
|---|---|
| `OPTIMAL` | 0 |
| `PRIMAL_INFEASIBLE` | 1 |
| `DUAL_INFEASIBLE` | 2 |
| `TIME_LIMIT` | 3 |
| `ITERATION_LIMIT` | 4 |
| `INFEASIBLE_OR_UNBOUNDED` | 5 |

Other statuses use code `-1`.

## Lower-Level Python API

`pdhcg._core.solve_once` accepts arrays directly and returns a result dictionary.
Its shorter argument names map to `Model` inputs as follows:

| `Model` input | `solve_once` argument |
|---|---|
| `objective_matrix`, `objective_matrix_low_rank` | `Q`, `R` |
| `objective_matrix_low_rank_middle` | `D` |
| `constraint_matrix` | `A` |
| `variable_cones` | `cones` |
| `affine_cone_matrix`, `affine_cone_offset` | `affine_F`, `affine_g` |

Other objective, bound, and `affine_cones` names are unchanged. `Q`, `R`, and
`A` are required arguments but may be `None`; warm starts use `primal_start`
and `dual_start`. Its `params` dictionary accepts the
[native parameter keys](parameters.md), and cone arguments use `ConeSpec`.
See [devices](../devices.md) for execution settings.

The returned dictionary contains `X`, `Pi`, `Status`, and `StatusCode`, plus
`PrimalObj`, `DualObj`, `ObjectiveGap`, `RelativeObjectiveGap`, `Iterations`,
`RuntimeSec`, `RescalingTimeSec`, `RelativePrimalResidual`,
`RelativeDualResidual`, and the four ray measures listed above.
