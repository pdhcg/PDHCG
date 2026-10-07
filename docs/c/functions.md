# C API Functions

## create_qp_problem

```c
qp_problem_t *create_qp_problem(
    const double *objective_c,
    const matrix_desc_t *Q_desc,
    const matrix_desc_t *R_desc,
    const matrix_desc_t *D_desc,
    const matrix_desc_t *A_desc,
    const double *con_lb, const double *con_ub,
    const double *var_lb, const double *var_ub,
    const double *objective_constant,
    int num_var_cones,
    const cone_spec_t *var_cones,
    const matrix_desc_t *affine_cone_matrix_desc,
    const double *affine_cone_offset,
    int num_affine_cones,
    const cone_spec_t *affine_cones
);
```

Creates a QP problem of the form
`min 0.5 * x^T (Q + R^T D R) x + c^T x + c0` subject to
`con_lb <= A x <= con_ub`, `F x + affine_cone_offset in K`,
`var_lb <= x <= var_ub`, and optional variable cone blocks. The affine cone
blocks must be disjoint and cover every row of `F`.
The [full Hessian must be positive semidefinite](../algorithm.md#standard-form).
Matrix descriptors accept the [formats listed in Types](types.md#matrix-format).
At least one of `A_desc`, `Q_desc`, `R_desc`, or `affine_cone_matrix_desc`
must be provided to define the number of variables.

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `objective_c` | Linear objective coefficients (size n); `NULL` means zero |
| `Q_desc` | Quadratic matrix, shape `n x n`; `NULL` means zero |
| `R_desc` | Low-rank factor, shape `k x n`; `NULL` omits the low-rank term |
| `D_desc` | Middle matrix in `R^T D R`, shape `k x k`; `NULL` means identity. May be indefinite. |
| `A_desc` | Scalar constraint matrix, shape `m x n`; `NULL` means no scalar rows |
| `con_lb` | Constraint lower bounds (size m); `NULL` means `-INFINITY` |
| `con_ub` | Constraint upper bounds (size m); `NULL` means `+INFINITY` |
| `var_lb` | Variable lower bounds (size n); `NULL` means `-INFINITY` |
| `var_ub` | Variable upper bounds (size n); `NULL` means `+INFINITY` |
| `objective_constant` | Constant `c0`; `NULL` means zero |
| `num_var_cones` | Number of variable cone blocks |
| `var_cones` | Array of variable `cone_spec_t` descriptors, or NULL when the count is zero |
| `affine_cone_matrix_desc` | Matrix `F` in the native affine cone constraint; NULL when no affine cones are present |
| `affine_cone_offset` | Offset vector with one entry per row of `F`; NULL means zero |
| `num_affine_cones` | Number of affine cone blocks covering `F` |
| `affine_cones` | Array of affine `cone_spec_t` descriptors, or NULL when the count is zero |

**Returns:** Pointer to allocated `qp_problem_t`, or NULL on error.
The problem owns copies of its inputs; caller arrays may be released after
creation. Free the problem with `qp_problem_free`.

---

## set_start_values

```c
void set_start_values(
    qp_problem_t *prob,
    const double *primal,
    const double *dual
);
```

Copies initial primal and dual solutions for warm starting. Passing `NULL` clears
the corresponding warm start, while values pinned by `set_cone_fixed` remain
part of the model and are preserved.

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `prob` | QP problem pointer |
| `primal` | Primal solution vector (size n, can be NULL) |
| `dual` | Dual solution vector (size `m + p`, ordered as `[dual_A, dual_F]`; can be NULL) |

Rejects a primal warm start that changes a value already pinned by
`set_cone_fixed`.

---

## set_cone_fixed

```c
int set_cone_fixed(
    qp_problem_t *prob,
    int cone_idx,
    int slot,
    double value
);
```

Pins one variable-cone coordinate to a finite `value`, preserved across warm
starts. Supports SOC, rotated SOC, exponential, and power cones; PSD blocks are
rejected. The solver checks that the resulting fixed section is nonempty before
preprocessing.

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `prob` | QP problem pointer |
| `cone_idx` | Cone index in `[0, num_cones)` |
| `slot` | Slot offset within the cone (0-based) |
| `value` | Fixed value |

**Returns:** 0 on success, nonzero on error (invalid indices, unsupported cone, or non-finite value).

---

## solve_qp_problem

```c
pdhcg_result_t *solve_qp_problem(
    const qp_problem_t *prob,
    const pdhg_parameters_t *params
);
```

Solves the problem without modifying it. See [Devices](../devices.md) for
backend selection and execution settings.

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `prob` | QP problem pointer |
| `params` | Solver parameters, or `NULL` for defaults |

**Returns:** Pointer to `pdhcg_result_t` containing solution information, or `NULL` on error.
The result owns its solution arrays independently of the problem; release it
with `pdhcg_result_free`.

---

## solve_qp_problem_distributed

```c
pdhcg_result_t *solve_qp_problem_distributed(
    const pdhg_parameters_t *params,
    const qp_problem_t *original_problem
);
```

Solves the QP problem using the distributed multi-GPU PDHCG algorithm.
The caller must initialize MPI with `MPI_Init` before this collective call;
all ranks participate, and only rank 0 supplies the problem.
See [Devices](../devices.md) for build and MPI execution requirements.

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `params` | Solver parameters (including `partition_method`, `permute_method`, `grid_size`, and `permute_block_size`) |
| `original_problem` | QP problem pointer (only required on rank 0; can be NULL on other ranks) |

**Returns:** Pointer to `pdhcg_result_t` on rank 0. Other ranks return `NULL`.
Returns `NULL` on error or when distributed support is unavailable.

---

## set_default_parameters

```c
void set_default_parameters(pdhg_parameters_t *params);
```

Fills the parameter struct with default values.

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `params` | Pointer to parameters struct to fill |

---

## pdhcg_validate_parameters

```c
int pdhcg_validate_parameters(
    const pdhg_parameters_t *params,
    char *error_message,
    size_t error_message_size
);
```

Validates solver parameter ranges and backend selection without starting a
solve. Returns `0` on success and nonzero on
failure. If `error_message` is non-`NULL` and its size is nonzero, the first
validation error is written there.

---

## pdhcg_result_free

```c
void pdhcg_result_free(pdhcg_result_t *results);
```

Frees the result and its solution arrays. Accepts `NULL`.

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `results` | Result pointer to free |

---

## qp_problem_free

```c
void qp_problem_free(qp_problem_t *prob);
```

Frees the problem and its owned data. Accepts `NULL`.

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `prob` | Problem pointer to free |
