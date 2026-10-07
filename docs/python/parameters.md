# Solver Parameters

## Setting Parameters

Use a Python alias or its native key with any of these interfaces:

```python
m.setParam("TimeLimit", 60)
m.setParams(OptimalityTol=1e-6, FeasibilityTol=1e-6)
m.Params.LogLevel = 0
m.Params["time_sec_limit"] = 120
print(m.getParam("TimeLimit"))
```

Unknown names and invalid values raise an error. Backend selection and CPU
thread configuration are documented under [devices](../devices.md).

## Parameter Reference

### Execution and Logging

| Parameter | Native key | Type | Default | Description |
|---|---|---|---|---|
| `Threads` | `num_threads` | int | 0 | [CPU thread limit](../devices.md#cpu-threads) |
| `LogLevel` | `verbose` | int | 1 | 0: silent; 1: summary; 2: iteration details |
| `Presolve` | `presolve` | bool | False | Enable PreFOS when built with presolve support |

`Verbosity`, `OutputFlag`, and `LogToConsole` are also aliases for `verbose`.

### Termination Criteria

| Parameter | Native key | Type | Default | Description |
|---|---|---|---|---|
| `TimeLimit` | `time_sec_limit` | float | 3600.0 | Time limit in seconds |
| `IterationLimit` | `iteration_limit` | int | 2147483647 | Maximum iteration count |
| `OptimalityTol` | `eps_optimal_relative` | float | 1e-4 | Relative optimality tolerance |
| `FeasibilityTol` | `eps_feasible_relative` | float | 1e-4 | Relative feasibility tolerance |
| `InfeasibleTol` | `eps_infeasible` | float | 1e-10 | Infeasibility certificate tolerance |
| `TermCheckFreq` | `termination_evaluation_frequency` | int | 200 | Iterations between termination checks |
| `OptimalityNorm` | `optimality_norm` | str | "linf" | Residual norm: "l2" or "linf" |
| — | `feasibility_polishing` | bool | False | Enable feasibility polishing |
| — | `eps_feas_polish_relative` | float | 1e-6 | Relative polishing tolerance |

### Scaling and Quadratic Updates

| Parameter | Native key | Type | Default | Description |
|---|---|---|---|---|
| `CurtisReidIters` | `curtis_reid_iterations` | int | 0 | Curtis–Reid scaling iterations; 0 disables it |
| `RuizIters` | `l_inf_ruiz_iterations` | int | 10 | L-inf Ruiz scaling iterations |
| `PCAlpha` | `pock_chambolle_alpha` | float | 1.0 | Pock–Chambolle scaling exponent |
| `UsePCAlpha` | `has_pock_chambolle_alpha` | bool | True | Enable Pock–Chambolle scaling |
| `BoundObjRescaling` | `bound_objective_rescaling` | bool | True | Rescale bounds and objective |
| `UseConePreservingScaling` | `use_cone_preserving_scaling` | bool | True | Use one scaling value per cone block |
| `ReflectionCoeff` | `reflection_coefficient` | float | 1.0 | Reflection coefficient |
| `NonDiagonalQuadraticMode` | `non_diagonal_quadratic_mode` | str | "inner" | Non-diagonal Q update: "inner" or "linearized" |

### Restarts

| Parameter | Native key | Type | Default | Description |
|---|---|---|---|---|
| `RestartArtificialThresh` | `artificial_restart_threshold` | float | 0.36 | Artificial restart threshold |
| `RestartSufficientReduction` | `sufficient_reduction_for_restart` | float | 0.2 | Sufficient reduction threshold |
| `RestartNecessaryReduction` | `necessary_reduction_for_restart` | float | 0.8 | Necessary reduction threshold |
| `RestartKp` | `k_p` | float | 0.99 | Proportional coefficient for primal-weight updates |

### Inner Solver

| Parameter | Native key | Type | Default | Description |
|---|---|---|---|---|
| `InnerIterLimit` | `inner_iter_limit` | int | 1000 | Maximum inner iteration count |
| `InnerInitTol` | `inner_init_tol` | float | 1e-3 | Initial inner tolerance |
| `InnerMinTol` | `inner_min_tol` | float | 1e-9 | Minimum inner tolerance |
| `DiagJacobiPrecond` | `diag_jacobi_precond` | bool | True | Jacobi diagonal preconditioner |

### Singular Value Estimation

| Parameter | Native key | Type | Default | Description |
|---|---|---|---|---|
| `SVMaxIter` | `sv_max_iter` | int | 5000 | Maximum power-method iteration count |
| `SVTol` | `sv_tol` | float | 1e-4 | Power-method tolerance |

Cone specifications are [model data](model.md#cone-constraints), not solver
parameters. The [lower-level Python API](model.md#lower-level-python-api)
accepts native keys in its `params` dictionary.
