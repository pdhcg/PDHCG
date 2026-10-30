/*
Copyright 2026 Hongpei Li

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#include "device_general_op.h"
#include "pdhcg.h"
#include "preprocess.h"
#include "solver_state.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Use the CPU backend as the reference, including when testing CUDA. */
qp_problem_t *pdhcg_cpu_convert_problem_to_device(const qp_problem_t *problem, grid_context_t *grid_context);
pdhg_solver_state_t *pdhcg_cpu_initialize_solver_state(const pdhg_parameters_t *params,
                                                       const qp_problem_t *problem,
                                                       const qp_problem_t *device_problem,
                                                       grid_context_t *grid_context);
void pdhcg_cpu_free_device_problem(qp_problem_t *problem);
void pdhcg_cpu_pdhg_solver_state_free(pdhg_solver_state_t *state);

static pdhg_solver_state_t *create_cpu_reference(const pdhg_parameters_t *params, const qp_problem_t *problem)
{
    pdhg_parameters_t initialize_params = *params;
    initialize_params.sv_max_iter = 2;
    qp_problem_t *device_problem = pdhcg_cpu_convert_problem_to_device(problem, NULL);
    pdhg_solver_state_t *state = pdhcg_cpu_initialize_solver_state(&initialize_params, problem, device_problem, NULL);
    pdhcg_cpu_free_device_problem(device_problem);
    return state;
}

static int failures;
static int cases;
static char case_name[96];

static void *allocate(size_t count, size_t size)
{
    void *memory = calloc(count ? count : 1, size);
    if (!memory)
        abort();
    return memory;
}

static void copy_to_host(void *destination, const void *source, size_t bytes, int device)
{
    if (!bytes)
        return;
    if (!source)
    {
        fprintf(stderr, "%s: missing buffer\n", case_name);
        abort();
    }
    if (device)
    {
        if (pdhcg_device_copy(destination, source, bytes, PDHCG_COPY_DEVICE_TO_HOST))
            abort();
    }
    else
        memcpy(destination, source, bytes);
}

static void check_close(const char *field, int index, double actual, double expected)
{
    if (actual == expected)
        return;
    if (!isfinite(actual) || !isfinite(expected) || fabs(actual - expected) > 2e-11 * fmax(1.0, fabs(expected)))
    {
        fprintf(stderr, "%s: %s[%d] got %.17g, expected %.17g\n", case_name, field, index, actual, expected);
        ++failures;
    }
}

static void check_array(const char *field, const double *actual, const double *expected, int length, int device)
{
    double *values = allocate((size_t)length, sizeof(double));
    copy_to_host(values, actual, (size_t)length * sizeof(double), device);
    for (int i = 0; i < length; ++i)
        check_close(field, i, values[i], expected[i]);
    free(values);
}

static void check_csr(const char *field,
                      const device_sparse_matrix_csr_t *actual,
                      const device_sparse_matrix_csr_t *expected,
                      int rows,
                      int nnz,
                      int device)
{
    if (!nnz)
        return;
    int *row_ptr = allocate((size_t)rows + 1, sizeof(int));
    int *col_ind = allocate((size_t)nnz, sizeof(int));
    copy_to_host(row_ptr, actual->row_ptr, ((size_t)rows + 1) * sizeof(int), device);
    copy_to_host(col_ind, actual->col_ind, (size_t)nnz * sizeof(int), device);
    if (memcmp(row_ptr, expected->row_ptr, ((size_t)rows + 1) * sizeof(int)) ||
        memcmp(col_ind, expected->col_ind, (size_t)nnz * sizeof(int)))
    {
        fprintf(stderr, "%s: %s CSR structure changed\n", case_name, field);
        ++failures;
    }
    check_array(field, actual->val, expected->val, nnz, device);
    free(row_ptr);
    free(col_ind);
}

typedef struct
{
    int row;
    int column;
    double value;
} matrix_entry_t;

static int compare_matrix_entries(const void *left, const void *right)
{
    const matrix_entry_t *a = left;
    const matrix_entry_t *b = right;
    if (a->row != b->row)
        return a->row < b->row ? -1 : 1;
    if (a->column != b->column)
        return a->column < b->column ? -1 : 1;
    return (a->value > b->value) - (a->value < b->value);
}

/* Check each transposed entry separately, including duplicate coordinates. */
static void check_transpose(const char *field,
                            const device_sparse_matrix_csr_t *transposed,
                            const device_sparse_matrix_csr_t *source,
                            int rows,
                            int columns,
                            int nnz)
{
    if (!nnz)
        return;
    int *row_ptr = allocate((size_t)columns + 1, sizeof(int));
    int *col_ind = allocate((size_t)nnz, sizeof(int));
    double *values = allocate((size_t)nnz, sizeof(double));
    matrix_entry_t *actual = allocate((size_t)nnz, sizeof(matrix_entry_t));
    matrix_entry_t *expected = allocate((size_t)nnz, sizeof(matrix_entry_t));
    copy_to_host(row_ptr, transposed->row_ptr, ((size_t)columns + 1) * sizeof(int), 1);
    copy_to_host(col_ind, transposed->col_ind, (size_t)nnz * sizeof(int), 1);
    copy_to_host(values, transposed->val, (size_t)nnz * sizeof(double), 1);
    if (row_ptr[0] != 0 || row_ptr[columns] != nnz)
        abort();
    for (int column = 0; column < columns; ++column)
    {
        if (row_ptr[column] > row_ptr[column + 1] || row_ptr[column] < 0 || row_ptr[column + 1] > nnz)
            abort();
        for (int k = row_ptr[column]; k < row_ptr[column + 1]; ++k)
        {
            int row = col_ind[k];
            if (row < 0 || row >= rows)
                abort();
            actual[k] = (matrix_entry_t){row, column, values[k]};
        }
    }
    for (int row = 0; row < rows; ++row)
        for (int k = source->row_ptr[row]; k < source->row_ptr[row + 1]; ++k)
            expected[k] = (matrix_entry_t){row, source->col_ind[k], source->val[k]};
    qsort(actual, (size_t)nnz, sizeof(matrix_entry_t), compare_matrix_entries);
    qsort(expected, (size_t)nnz, sizeof(matrix_entry_t), compare_matrix_entries);
    for (int k = 0; k < nnz; ++k)
    {
        if (actual[k].row != expected[k].row || actual[k].column != expected[k].column)
        {
            fprintf(stderr, "%s: %s transposed coordinate mismatch at entry %d\n", case_name, field, k);
            ++failures;
        }
        check_close(field, k, actual[k].value, expected[k].value);
    }
    free(row_ptr);
    free(col_ind);
    free(values);
    free(actual);
    free(expected);
}

static void compare_scaling(const qp_problem_t *problem, const pdhg_parameters_t *params)
{
    ++cases;
    if (!problem)
    {
        fprintf(stderr, "%s: failed to construct problem\n", case_name);
        ++failures;
        return;
    }
    pdhg_solver_state_t *reference = create_cpu_reference(params, problem);
    pdhg_parameters_t initialize_params = *params;
    initialize_params.sv_max_iter = 2;
    qp_problem_t *device_actual = convert_problem_to_device(problem, NULL);
    if ((device_actual->primal_start == NULL) != (problem->primal_start == NULL) ||
        (device_actual->dual_start == NULL) != (problem->dual_start == NULL) ||
        (device_actual->objective_sparse_matrix == NULL) != (problem->objective_sparse_matrix == NULL) ||
        (device_actual->objective_lowrank_matrix == NULL) != (problem->objective_lowrank_matrix == NULL) ||
        (device_actual->objective_lowrank_middle_matrix == NULL) != (problem->objective_lowrank_middle_matrix == NULL))
    {
        fprintf(stderr, "%s: conversion changed optional field presence\n", case_name);
        ++failures;
    }
    pdhg_solver_state_t *actual = initialize_solver_state(&initialize_params, problem, device_actual, NULL);
    free_device_problem(device_actual);
    if (!reference || !actual || !actual->constraint_matrix)
    {
        fprintf(stderr, "%s: scaling returned no problem\n", case_name);
        ++failures;
        goto cleanup;
    }
    const int n = problem->num_variables;
    const int m = problem->num_constraints;
    const int rank = problem->num_rank_lowrank_obj;
    if (actual->num_variables != n || actual->num_constraints != m ||
        actual->quadratic_objective_term->quad_obj_type != reference->quadratic_objective_term->quad_obj_type)
    {
        fprintf(stderr, "%s: problem metadata changed\n", case_name);
        ++failures;
    }
    check_array("row scaling", actual->constraint_rescaling, reference->constraint_rescaling, m, 1);
    check_array("column scaling", actual->variable_rescaling, reference->variable_rescaling, n, 1);
    check_close("bound scaling", 0, actual->constraint_bound_rescaling, reference->constraint_bound_rescaling);
    check_close("objective scaling", 0, actual->objective_vector_rescaling, reference->objective_vector_rescaling);
    check_close("objective constant", 0, actual->objective_constant, reference->objective_constant);
    if (!isfinite(actual->rescaling_time_sec) || actual->rescaling_time_sec < 0.0)
    {
        fprintf(stderr, "%s: invalid scaling duration\n", case_name);
        ++failures;
    }
#define CHECK_VECTOR(field, length) check_array(#field, actual->field, reference->field, length, 1)
    CHECK_VECTOR(objective_vector, n);
    CHECK_VECTOR(variable_lower_bound, n);
    CHECK_VECTOR(variable_upper_bound, n);
    CHECK_VECTOR(constraint_lower_bound, m);
    CHECK_VECTOR(constraint_upper_bound, m);
    CHECK_VECTOR(affine_cone_offset, m);
#undef CHECK_VECTOR
    check_csr(
        "A", actual->constraint_matrix, reference->constraint_matrix, m, problem->constraint_matrix_num_nonzeros, 1);
    if (!problem->constraint_matrix)
    {
        const device_sparse_matrix_csr_t *empty[] = {actual->constraint_matrix, actual->constraint_matrix_t};
        for (int matrix = 0; matrix < 2; ++matrix)
        {
            int count = empty[matrix]->num_rows + 1;
            int *rows = allocate((size_t)count, sizeof(int));
            copy_to_host(rows, empty[matrix]->row_ptr, (size_t)count * sizeof(int), 1);
            check_close("empty CSR nnz", matrix, empty[matrix]->num_nonzeros, 0);
            for (int row = 0; row < count; ++row)
                check_close("empty CSR row offset", row, rows[row], 0);
            free(rows);
        }
    }
    if (actual->quadratic_objective_term->quad_obj_type != PDHCG_DIAG_Q)
    {
        check_csr("Q",
                  actual->quadratic_objective_term->objective_sparse_matrix,
                  reference->quadratic_objective_term->objective_sparse_matrix,
                  n,
                  problem->objective_sparse_matrix_num_nonzeros,
                  1);
        check_transpose("Q symmetry after scaling",
                        actual->quadratic_objective_term->objective_sparse_matrix,
                        reference->quadratic_objective_term->objective_sparse_matrix,
                        n,
                        n,
                        problem->objective_sparse_matrix_num_nonzeros);
    }
    else
    {
        check_array("diagonal Q reference",
                    actual->quadratic_objective_term->diagonal_objective_matrix,
                    reference->quadratic_objective_term->diagonal_objective_matrix,
                    n,
                    1);
        /* The resident state has discarded diagonal Q's CSR. Independently
           recover its last-entry-wins values from the original data and scaling. */
        double *diagonal = allocate((size_t)n, sizeof(double));
        const CsrComponent *q = problem->objective_sparse_matrix;
        long double multiplier =
            (long double)reference->objective_vector_rescaling / reference->constraint_bound_rescaling;
        for (int row = 0; row < n; ++row)
            for (int j = q->row_ptr[row]; j < q->row_ptr[row + 1]; ++j)
                diagonal[row] = (double)((long double)q->val[j] / reference->variable_rescaling[row] /
                                         reference->variable_rescaling[q->col_ind[j]] * multiplier) +
                    1e-12;
        check_array("diagonal Q", actual->quadratic_objective_term->diagonal_objective_matrix, diagonal, n, 1);
        if (actual->quadratic_objective_term->objective_sparse_matrix)
        {
            fprintf(stderr, "%s: diagonal Q retains an unused CSR copy\n", case_name);
            ++failures;
        }
        free(diagonal);
    }
    check_csr("R",
              actual->quadratic_objective_term->objective_lowrank_matrix,
              reference->quadratic_objective_term->objective_lowrank_matrix,
              rank,
              problem->objective_lowrank_matrix_num_nonzeros,
              1);
    check_transpose("A transpose",
                    actual->constraint_matrix_t,
                    reference->constraint_matrix,
                    m,
                    n,
                    problem->constraint_matrix_num_nonzeros);
    check_transpose("R transpose",
                    actual->quadratic_objective_term->objective_lowrank_matrix_t,
                    reference->quadratic_objective_term->objective_lowrank_matrix,
                    rank,
                    n,
                    problem->objective_lowrank_matrix_num_nonzeros);

    lowrank_middle_kind_t middle_type = PDHCG_D_NONE;
    if (problem->objective_lowrank_middle_matrix_num_nonzeros > 0)
    {
        const CsrComponent *middle = problem->objective_lowrank_middle_matrix;
        middle_type = PDHCG_D_DIAG;
        double *dense = allocate((size_t)rank * rank, sizeof(double));
        double *diagonal = allocate((size_t)rank, sizeof(double));
        for (int row = 0; row < rank; ++row)
            for (int j = middle->row_ptr[row]; j < middle->row_ptr[row + 1]; ++j)
            {
                int column = middle->col_ind[j];
                dense[row * rank + column] = middle->val[j];
                if (column != row)
                    middle_type = PDHCG_D_DENSE;
                else
                    diagonal[row] = middle->val[j];
            }
        if (middle_type == PDHCG_D_DIAG)
            check_array("middle diagonal", actual->quadratic_objective_term->d_middle_diag, diagonal, rank, 1);
        else
            check_array("middle dense", actual->quadratic_objective_term->d_middle_dense, dense, rank * rank, 1);
        free(dense);
        free(diagonal);
    }
    if (actual->quadratic_objective_term->lowrank_middle_type != middle_type)
    {
        fprintf(stderr, "%s: low-rank middle representation changed\n", case_name);
        ++failures;
    }
    if (problem->primal_start && problem->cones.num_cones == 0)
    {
        double *start = allocate((size_t)n, sizeof(double));
        for (int j = 0; j < n; ++j)
        {
            double value =
                problem->primal_start[j] * reference->variable_rescaling[j] * reference->constraint_bound_rescaling;
            start[j] = fmin(reference->variable_upper_bound[j], fmax(reference->variable_lower_bound[j], value));
        }
        check_array("primal start", actual->initial_primal_solution, start, n, 1);
        free(start);
    }
    if (problem->dual_start)
    {
        double *start = allocate((size_t)m, sizeof(double));
        for (int j = 0; j < m; ++j)
            start[j] =
                problem->dual_start[j] * reference->constraint_rescaling[j] * reference->objective_vector_rescaling;
        check_array("dual start", actual->initial_dual_solution, start, m, 1);
        free(start);
    }

cleanup:
    pdhcg_cpu_pdhg_solver_state_free(reference);
    pdhg_solver_state_free(actual);
}

static matrix_desc_t dense_matrix(int rows, int columns, const double *values)
{
    matrix_desc_t matrix = {.m = rows, .n = columns, .fmt = matrix_dense};
    matrix.data.dense.A = values;
    return matrix;
}

/* Empty row and column, mixed bounds, nonzero starts and every objective form. */
static qp_problem_t *make_quadratic_problem(int kind)
{
    const double a[] = {0.001, -30, 0, 2, 0, 0, 0, 0, 0, 0, 70, 0.02, 0, -0.4, 0};
    double q[25] = {0};
    const double r[] = {3, -0.1, 0, 4, 0, 0.03, 2, -0.5, 0, 7};
    double d[] = {2, 0, 0, 0.5};
    const double c[] = {3, -7, 0, 2, 0};
    const double lower[] = {-INFINITY, 0, -8};
    const double upper[] = {5, 0, INFINITY};
    const double var_lower[] = {-2, -INFINITY, 0, -INFINITY, -INFINITY};
    const double var_upper[] = {9, INFINITY, 0, 4, INFINITY};
    const double primal[] = {0.5, -2, 0, 1, 0.3};
    const double dual[] = {0.2, 0, -1};
    const double constant = 1.75;
    for (int j = 0; j < 5; ++j)
        q[j * 5 + j] = j == 2 ? 0 : pow(3.0, j);
    if (kind >= 2)
        q[1] = q[5] = -0.5;
    if (kind == 5 || kind == 6)
        d[1] = d[2] = -0.2;
    matrix_desc_t A = dense_matrix(3, 5, a);
    matrix_desc_t Q = dense_matrix(5, 5, q);
    matrix_desc_t R = dense_matrix(2, 5, r);
    matrix_desc_t D = dense_matrix(2, 2, d);
    qp_problem_t *problem = create_qp_problem(c,
                                              (kind == 1 || kind == 2 || kind == 6) ? &Q : NULL,
                                              kind >= 3 ? &R : NULL,
                                              kind >= 4 ? &D : NULL,
                                              &A,
                                              lower,
                                              upper,
                                              var_lower,
                                              var_upper,
                                              &constant,
                                              0,
                                              NULL,
                                              NULL,
                                              NULL,
                                              0,
                                              NULL);
    if (problem)
        set_start_values(problem, primal, dual);
    return problem;
}

static qp_problem_t *make_cone_problem(void)
{
    enum
    {
        N = 26,
        P = 24,
        M = 3
    };
    double a[M * N] = {0}, f[P * N] = {0}, q[N * N] = {0};
    double c[N], primal[N], dual[M + P], offset[P];
    const double lower[] = {-INFINITY, 0, -3};
    const double upper[] = {7, 0, 4};
    const char fixed[] = {0, 1, 0};
    cone_spec_t variable[] = {
        {.type = CONE_ROTATED_SOC, .start_idx = 2, .v_dim = 1, .is_fixed = fixed},
        {.type = CONE_STANDARD_SOC, .start_idx = 5, .v_dim = 7},
        {.type = CONE_EXPONENTIAL, .start_idx = 14, .v_dim = 1, .is_fixed = fixed},
        {.type = CONE_POWER, .start_idx = 17, .v_dim = 1, .power_alpha = 0.3, .is_fixed = fixed},
        {.type = CONE_PSD, .start_idx = 20, .v_dim = 3},
    };
    cone_spec_t affine[5];
    memcpy(affine, variable, sizeof(affine));
    for (int i = 0; i < 5; ++i)
    {
        affine[i].start_idx -= 2;
        affine[i].is_fixed = NULL;
    }
    for (int j = 0; j < N; ++j)
    {
        c[j] = (j % 5 - 2) * 0.3;
        primal[j] = (j + 1) * 0.25;
        q[j * N + j] = (j + 1) * 0.5;
        a[j] = ldexp((j % 2 ? -1 : 1), j % 15 - 7);
        a[2 * N + j] = (j % 3 ? 0 : j + 1);
    }
    a[1] = a[2 * N + 1] = 0;
    for (int row = 0; row < P; ++row)
    {
        f[row * N + row + 2] = ldexp(1.0, row % 11 - 5);
        f[row * N] = (row % 2 ? -0.02 : 1.5);
        offset[row] = (row % 7 - 3) * 0.4;
    }
    for (int row = 0; row < M + P; ++row)
        dual[row] = (row % 5 - 2) * 0.2;
    matrix_desc_t A = dense_matrix(M, N, a);
    matrix_desc_t F = dense_matrix(P, N, f);
    matrix_desc_t Q = dense_matrix(N, N, q);
    qp_problem_t *problem =
        create_qp_problem(c, &Q, NULL, NULL, &A, lower, upper, NULL, NULL, NULL, 5, variable, &F, offset, 5, affine);
    if (problem)
        set_start_values(problem, primal, dual);
    return problem;
}

/* A row crossing several CUDA blocks and repeated coordinates in both orders. */
static qp_problem_t *make_long_row_problem(void)
{
    enum
    {
        N = 1031,
        NNZ = N + 3
    };
    int rows[] = {0, N + 1, N + 1, NNZ};
    int *columns = allocate(NNZ, sizeof(int));
    double *values = allocate(NNZ, sizeof(double));
    double *objective = allocate(N, sizeof(double));
    for (int j = 0; j < N; ++j)
    {
        columns[j] = N - j - 1;
        values[j] = ldexp((j % 2 ? -1.0 : 1.0), j % 25 - 12);
        objective[j] = (j % 13 - 6) * 0.1;
    }
    columns[N] = 23;
    values[N] = -0.125;
    columns[N + 1] = columns[N + 2] = 17;
    values[N + 1] = 32;
    values[N + 2] = -4;
    const double bounds[] = {3, 0, -2};
    matrix_desc_t A = {.m = 3, .n = N, .fmt = matrix_csr};
    A.data.csr.nnz = NNZ;
    A.data.csr.row_ptr = rows;
    A.data.csr.col_ind = columns;
    A.data.csr.vals = values;
    qp_problem_t *problem = create_qp_problem(
        objective, NULL, NULL, NULL, &A, bounds, bounds, NULL, NULL, NULL, 0, NULL, NULL, NULL, 0, NULL);
    free(columns);
    free(values);
    free(objective);
    return problem;
}

static pdhg_parameters_t scaling_parameters(int method)
{
    pdhg_parameters_t params;
    set_default_parameters(&params);
    params.verbose = 0;
    params.curtis_reid_iterations = method == 1 || method == 9 ? 7 : 0;
    params.l_inf_ruiz_iterations = method == 2 || method == 9 ? 4 : 0;
    params.has_pock_chambolle_alpha = (method >= 3 && method <= 7) || method == 9;
    params.pock_chambolle_alpha = method == 9 ? 1.0 : (method - 3) * 0.5;
    params.bound_objective_rescaling = method >= 8;
    return params;
}

static void test_large_cone(int psd, int affine)
{
    const int order = psd ? 257 : 1029;
    const int length = psd ? order * (order + 1) / 2 : order + 2;
    const int rows = affine ? length : 1;
    const int columns = affine ? 1 : length;
    int *row_ptr = allocate((size_t)rows + 1, sizeof(int));
    int *col_ind = allocate((size_t)length, sizeof(int));
    double *values = allocate((size_t)length, sizeof(double));
    double *objective = allocate((size_t)columns, sizeof(double));
    double *offset = affine ? allocate((size_t)length, sizeof(double)) : NULL;
    for (int row = 0; row <= rows; ++row)
        row_ptr[row] = affine ? row : row * length;
    for (int k = 0; k < length; ++k)
    {
        col_ind[k] = affine ? 0 : k;
        values[k] = ldexp(k % 2 ? -1.0 : 1.0, k % 19 - 9);
        if (affine)
            offset[k] = (k % 7 - 3) * 0.1;
    }
    matrix_desc_t matrix = {.m = rows, .n = columns, .fmt = matrix_csr};
    matrix.data.csr.nnz = length;
    matrix.data.csr.row_ptr = row_ptr;
    matrix.data.csr.col_ind = col_ind;
    matrix.data.csr.vals = values;
    cone_spec_t cone = {.type = psd ? CONE_PSD : CONE_STANDARD_SOC, .start_idx = 0, .v_dim = order};
    qp_problem_t *problem = create_qp_problem(objective,
                                              NULL,
                                              NULL,
                                              NULL,
                                              affine ? NULL : &matrix,
                                              NULL,
                                              NULL,
                                              NULL,
                                              NULL,
                                              NULL,
                                              affine ? 0 : 1,
                                              affine ? NULL : &cone,
                                              affine ? &matrix : NULL,
                                              offset,
                                              affine ? 1 : 0,
                                              affine ? &cone : NULL);
    free(row_ptr);
    free(col_ind);
    free(values);
    free(objective);
    free(offset);
    snprintf(case_name,
             sizeof(case_name),
             "%s large %s order=%d",
             affine ? "affine" : "variable",
             psd ? "PSD" : "SOC",
             order);
    pdhg_parameters_t params = scaling_parameters(9);
    params.use_cone_preserving_scaling = !psd;
    compare_scaling(problem, &params);
    qp_problem_free(problem);
}

static void test_unconstrained_quadratic(void)
{
    const double q[] = {4, -1, -1, 2};
    const double c[] = {1, -3};
    matrix_desc_t Q = dense_matrix(2, 2, q);
    const int row_ptr[] = {0};
    matrix_desc_t A = {.m = 0, .n = 2, .fmt = matrix_csr};
    A.data.csr.row_ptr = row_ptr;
    qp_problem_t *problem =
        create_qp_problem(c, &Q, NULL, NULL, &A, NULL, NULL, NULL, NULL, NULL, 0, NULL, NULL, NULL, 0, NULL);
    snprintf(case_name, sizeof(case_name), "unconstrained quadratic with zero-row A");
    pdhg_parameters_t params = scaling_parameters(9);
    params.non_diagonal_quadratic_mode = NON_DIAGONAL_QUADRATIC_LINEARIZED;
    compare_scaling(problem, &params);
    qp_problem_free(problem);
}

static void test_empty_matrix(void)
{
    const int rows[] = {0, 0, 0};
    const double c[] = {0, 3, -2};
    matrix_desc_t A = {.m = 2, .n = 3, .fmt = matrix_csr};
    A.data.csr.row_ptr = rows;
    for (int constraints = 0; constraints <= 2; constraints += 2)
    {
        A.m = constraints;
        qp_problem_t *problem =
            create_qp_problem(c, NULL, NULL, NULL, &A, NULL, NULL, NULL, NULL, NULL, 0, NULL, NULL, NULL, 0, NULL);
        for (int method = 0; method <= 9; ++method)
        {
            snprintf(case_name, sizeof(case_name), "empty A rows=%d method=%d", constraints, method);
            pdhg_parameters_t params = scaling_parameters(method);
            compare_scaling(problem, &params);
        }
        if (problem && constraints > 0)
        {
            CsrComponent *empty = problem->constraint_matrix;
            problem->constraint_matrix = NULL;
            snprintf(case_name, sizeof(case_name), "NULL A with nonzero constraint rows");
            pdhg_parameters_t params = scaling_parameters(9);
            compare_scaling(problem, &params);
            problem->constraint_matrix = empty;
        }
        qp_problem_free(problem);
    }
}

static void check_device_arrays(const char *field, const double *actual, const double *reference, int length)
{
    double *expected = allocate((size_t)length, sizeof(double));
    copy_to_host(expected, reference, (size_t)length * sizeof(double), 1);
    check_array(field, actual, expected, length, 1);
    free(expected);
}

static void check_state_matrix(const char *field,
                               const device_sparse_matrix_csr_t *actual,
                               const device_sparse_matrix_csr_t *reference)
{
    if (!actual || !reference)
    {
        if (actual != reference)
        {
            fprintf(stderr, "%s: %s presence differs\n", case_name, field);
            ++failures;
        }
        return;
    }
    check_array(field, actual->val, reference->val, reference->num_nonzeros, 1);
}

static qp_problem_t *make_fixed_cone_problem(void)
{
    const double a[] = {0.01, -3, 10};
    const double c[] = {0.5, 0, 1};
    const double primal[] = {0.25, 1, 2};
    const double dual[] = {0.1};
    const double lower[] = {-INFINITY};
    const double upper[] = {5};
    const char fixed[] = {0, 1, 0};
    const cone_spec_t cone = {.type = CONE_STANDARD_SOC, .start_idx = 0, .v_dim = 1, .is_fixed = fixed};
    matrix_desc_t A = dense_matrix(1, 3, a);
    qp_problem_t *problem =
        create_qp_problem(c, NULL, NULL, NULL, &A, lower, upper, NULL, NULL, NULL, 1, &cone, NULL, NULL, 0, NULL);
    if (problem)
        set_start_values(problem, primal, dual);
    return problem;
}

static void test_initialization(int kind)
{
    snprintf(case_name, sizeof(case_name), "state initialization Q=%d", kind);
    qp_problem_t *problem = kind < 0 ? make_fixed_cone_problem() : make_quadratic_problem(kind);
    if (!problem)
        abort();
    ++cases;
    pdhg_parameters_t params = scaling_parameters(9);
    params.non_diagonal_quadratic_mode = NON_DIAGONAL_QUADRATIC_LINEARIZED;
    pdhg_solver_state_t *reference = create_cpu_reference(&params, problem);
    const int n = problem->num_variables;
    const int m = problem->num_constraints;
    double *expected_dual = allocate((size_t)m, sizeof(double));
    double *expected_primal = kind == 0 ? allocate((size_t)n, sizeof(double)) : NULL;
    for (int j = 0; j < m; ++j)
        expected_dual[j] =
            problem->dual_start[j] * reference->constraint_rescaling[j] * reference->objective_vector_rescaling;
    for (int j = 0; expected_primal && j < n; ++j)
        expected_primal[j] =
            problem->primal_start[j] * reference->variable_rescaling[j] * reference->constraint_bound_rescaling;
    qp_problem_t *device_actual = convert_problem_to_device(problem, NULL);
    pdhg_solver_state_t *actual = initialize_solver_state(&params, problem, device_actual, NULL);
    free_device_problem(device_actual);
    if (!reference || !actual)
        abort();
    /* Neither state may retain numerical arrays borrowed from these host owners. */
    qp_problem_free(problem);
    check_array("preserved initial dual start", actual->initial_dual_solution, expected_dual, m, 1);
    check_array("preserved current dual start", actual->current_dual_solution, expected_dual, m, 1);
    if (expected_primal)
    {
        check_array("preserved initial primal start", actual->initial_primal_solution, expected_primal, n, 1);
        check_array("preserved current primal start", actual->current_primal_solution, expected_primal, n, 1);
    }
    free(expected_dual);
    free(expected_primal);
#define CHECK_STATE_VECTOR(field, length) check_array(#field, actual->field, reference->field, length, 1)
    CHECK_STATE_VECTOR(objective_vector, n);
    CHECK_STATE_VECTOR(variable_lower_bound, n);
    CHECK_STATE_VECTOR(variable_upper_bound, n);
    CHECK_STATE_VECTOR(constraint_lower_bound, m);
    CHECK_STATE_VECTOR(constraint_upper_bound, m);
    CHECK_STATE_VECTOR(affine_cone_offset, m);
    CHECK_STATE_VECTOR(variable_lower_bound_finite_val, n);
    CHECK_STATE_VECTOR(variable_upper_bound_finite_val, n);
    CHECK_STATE_VECTOR(initial_primal_solution, n);
    CHECK_STATE_VECTOR(current_primal_solution, n);
    CHECK_STATE_VECTOR(constraint_lower_bound_finite_val, m);
    CHECK_STATE_VECTOR(constraint_upper_bound_finite_val, m);
    CHECK_STATE_VECTOR(initial_dual_solution, m);
    CHECK_STATE_VECTOR(current_dual_solution, m);
    CHECK_STATE_VECTOR(variable_rescaling, n);
    CHECK_STATE_VECTOR(constraint_rescaling, m);
#undef CHECK_STATE_VECTOR
    check_close("objective constant", 0, actual->objective_constant, reference->objective_constant);
    check_close("objective vector norm", 0, actual->objective_vector_norm, reference->objective_vector_norm);
    check_close("constraint bound norm", 0, actual->constraint_bound_norm, reference->constraint_bound_norm);
    check_state_matrix("state A", actual->constraint_matrix, reference->constraint_matrix);
    check_state_matrix("state A transpose", actual->constraint_matrix_t, reference->constraint_matrix_t);
    if (kind == 1)
        check_array("state diagonal Q",
                    actual->quadratic_objective_term->diagonal_objective_matrix,
                    reference->quadratic_objective_term->diagonal_objective_matrix,
                    n,
                    1);
    if (kind == 6)
    {
        const quadratic_objective_term_t *aq = actual->quadratic_objective_term;
        const quadratic_objective_term_t *rq = reference->quadratic_objective_term;
        check_state_matrix("state Q", aq->objective_sparse_matrix, rq->objective_sparse_matrix);
        check_state_matrix("state R", aq->objective_lowrank_matrix, rq->objective_lowrank_matrix);
        check_state_matrix("state R transpose", aq->objective_lowrank_matrix_t, rq->objective_lowrank_matrix_t);
        check_array("state middle D", aq->d_middle_dense, rq->d_middle_dense, 4, 1);
    }
    pdhcg_cpu_pdhg_solver_state_free(reference);
    pdhg_solver_state_free(actual);
}

/* In INNER mode Q/R do not affect equilibration; apply their final scaling once. */
static void test_deferred_quadratic(void)
{
    snprintf(case_name, sizeof(case_name), "deferred inner quadratic scaling");
    ++cases;
    qp_problem_t *lp = make_quadratic_problem(0);
    qp_problem_t *qp = make_quadratic_problem(6);
    if (!lp || !qp)
        abort();
    pdhg_parameters_t params = scaling_parameters(9);
    params.non_diagonal_quadratic_mode = NON_DIAGONAL_QUADRATIC_INNER;
    qp_problem_t *device_linear = convert_problem_to_device(lp, NULL);
    pdhg_solver_state_t *linear = initialize_solver_state(&params, lp, device_linear, NULL);
    free_device_problem(device_linear);
    qp_problem_t *device_quadratic = convert_problem_to_device(qp, NULL);
    pdhg_solver_state_t *quadratic = initialize_solver_state(&params, qp, device_quadratic, NULL);
    free_device_problem(device_quadratic);
    check_device_arrays("Q-independent column scaling", quadratic->variable_rescaling, linear->variable_rescaling, 5);
    check_device_arrays("Q-independent row scaling", quadratic->constraint_rescaling, linear->constraint_rescaling, 3);
    check_close(
        "Q-independent bound scaling", 0, quadratic->constraint_bound_rescaling, linear->constraint_bound_rescaling);
    check_close("Q-independent objective scaling",
                0,
                quadratic->objective_vector_rescaling,
                linear->objective_vector_rescaling);
    double scaling[5];
    copy_to_host(scaling, quadratic->variable_rescaling, sizeof(scaling), 1);
    double objective_scale = quadratic->objective_vector_rescaling / quadratic->constraint_bound_rescaling;
    const CsrComponent *matrices[] = {qp->objective_sparse_matrix, qp->objective_lowrank_matrix};
    const device_sparse_matrix_csr_t *scaled[] = {quadratic->quadratic_objective_term->objective_sparse_matrix,
                                                  quadratic->quadratic_objective_term->objective_lowrank_matrix};
    for (int kind = 0; kind < 2; ++kind)
    {
        const CsrComponent *matrix = matrices[kind];
        const int rows = kind == 0 ? 5 : 2;
        const int nonzeros = matrix->row_ptr[rows];
        double *expected = allocate((size_t)nonzeros, sizeof(double));
        for (int row = 0; row < rows; ++row)
            for (int j = matrix->row_ptr[row]; j < matrix->row_ptr[row + 1]; ++j)
            {
                int column = matrix->col_ind[j];
                expected[j] = kind == 0 ? matrix->val[j] * objective_scale / (scaling[row] * scaling[column])
                                        : matrix->val[j] * sqrt(objective_scale) / scaling[column];
            }
        check_array(kind == 0 ? "Q final scaling" : "R final scaling", scaled[kind]->val, expected, nonzeros, 1);
        if (kind == 1)
        {
            device_sparse_matrix_csr_t expected_r = {
                .row_ptr = matrix->row_ptr, .col_ind = matrix->col_ind, .val = expected};
            check_transpose("R transpose final scaling",
                            quadratic->quadratic_objective_term->objective_lowrank_matrix_t,
                            &expected_r,
                            rows,
                            5,
                            nonzeros);
        }
        free(expected);
    }
    pdhg_solver_state_free(linear);
    pdhg_solver_state_free(quadratic);
    qp_problem_free(lp);
    qp_problem_free(qp);
}

static void test_diagonal_regularization(void)
{
    qp_problem_t *problem = make_quadratic_problem(1);
    if (!problem)
        abort();
    CsrComponent *q = problem->objective_sparse_matrix;
    /* Keep row 2 absent, but retain an explicitly stored zero in row 0. */
    const double raw[] = {0.0, 2e-24, 4e-12, 1e-200};
    memcpy(q->val, raw, sizeof(raw));
    CsrComponent *a = problem->constraint_matrix;
    for (int j = 0; j < problem->constraint_matrix_num_nonzeros; ++j)
        a->val[j] *= a->col_ind[j] == 1 ? 1e24 : 1e-6;
    const int methods[] = {0, 2, 9};
    for (int method = 0; method < 3; ++method)
    {
        snprintf(case_name, sizeof(case_name), "diagonal regularization after scaling method=%d", methods[method]);
        ++cases;
        pdhg_parameters_t params = scaling_parameters(methods[method]);
        params.sv_max_iter = 2;
        qp_problem_t *device = convert_problem_to_device(problem, NULL);
        pdhg_solver_state_t *state = initialize_solver_state(&params, problem, device, NULL);
        free_device_problem(device);
        double factors[5], diagonal[5];
        copy_to_host(factors, state->variable_rescaling, sizeof(factors), 1);
        copy_to_host(diagonal, state->quadratic_objective_term->diagonal_objective_matrix, sizeof(diagonal), 1);
        long double multiplier = (long double)state->objective_vector_rescaling / state->constraint_bound_rescaling;
        for (int row = 0; row < 5; ++row)
        {
            double expected = q->row_ptr[row] == q->row_ptr[row + 1]
                ? 0.0
                : (double)((long double)q->val[q->row_ptr[row]] / factors[row] / factors[row] * multiplier) + 1e-12;
            /* The usual absolute tolerance would hide the entire regularizer. */
            if (!isfinite(diagonal[row]) || !isfinite(expected) ||
                fabs(diagonal[row] - expected) > 2e-13 * fabs(expected))
            {
                fprintf(
                    stderr, "%s: diagonal[%d] got %.17g, expected %.17g\n", case_name, row, diagonal[row], expected);
                ++failures;
            }
        }
        if (methods[method] == 2 && factors[1] < 1e6)
        {
            fprintf(stderr, "%s: fixture did not exercise a large scaling factor\n", case_name);
            ++failures;
        }
        pdhg_solver_state_free(state);
    }
    qp_problem_free(problem);
}

static void test_preparation_ownership(void)
{
    snprintf(case_name, sizeof(case_name), "state owns storage after converted and host sources are freed");
    ++cases;
    qp_problem_t *source = make_quadratic_problem(6);
    qp_problem_t *original = make_quadratic_problem(6);
    if (!source || !original)
        abort();
    pdhg_parameters_t params = scaling_parameters(9);
    pdhg_solver_state_t *reference = create_cpu_reference(&params, source);
    qp_problem_t *device_prepared = convert_problem_to_device(source, NULL);
    pdhg_solver_state_t *prepared = initialize_solver_state(&params, source, device_prepared, NULL);
    const double *host_vectors[] = {source->variable_lower_bound,
                                    source->variable_upper_bound,
                                    source->objective_vector,
                                    source->constraint_lower_bound,
                                    source->constraint_upper_bound,
                                    source->affine_cone_offset,
                                    source->primal_start,
                                    source->dual_start};
    const double *original_vectors[] = {original->variable_lower_bound,
                                        original->variable_upper_bound,
                                        original->objective_vector,
                                        original->constraint_lower_bound,
                                        original->constraint_upper_bound,
                                        original->affine_cone_offset,
                                        original->primal_start,
                                        original->dual_start};
    const double *device_vectors[] = {device_prepared->variable_lower_bound,
                                      device_prepared->variable_upper_bound,
                                      device_prepared->objective_vector,
                                      device_prepared->constraint_lower_bound,
                                      device_prepared->constraint_upper_bound,
                                      device_prepared->affine_cone_offset,
                                      device_prepared->primal_start,
                                      device_prepared->dual_start};
    const double *state_vectors[] = {prepared->variable_lower_bound,
                                     prepared->variable_upper_bound,
                                     prepared->objective_vector,
                                     prepared->constraint_lower_bound,
                                     prepared->constraint_upper_bound,
                                     prepared->affine_cone_offset,
                                     prepared->initial_primal_solution,
                                     prepared->initial_dual_solution};
    const int lengths[] = {5, 5, 5, 3, 3, 3, 5, 3};
    for (int field = 0; field < 8; ++field)
    {
        if (device_vectors[field] == host_vectors[field] || state_vectors[field] == device_vectors[field])
        {
            fprintf(stderr, "%s: vector %d aliases its source\n", case_name, field);
            ++failures;
        }
        check_array("unchanged source vector", host_vectors[field], original_vectors[field], lengths[field], 0);
        check_array("unchanged converted vector", device_vectors[field], original_vectors[field], lengths[field], 1);
    }
    const CsrComponent *raw_matrices[] = {
        source->constraint_matrix, source->objective_sparse_matrix, source->objective_lowrank_matrix};
    const CsrComponent *original_matrices[] = {
        original->constraint_matrix, original->objective_sparse_matrix, original->objective_lowrank_matrix};
    const CsrComponent *converted_matrices[] = {device_prepared->constraint_matrix,
                                                device_prepared->objective_sparse_matrix,
                                                device_prepared->objective_lowrank_matrix};
    const device_sparse_matrix_csr_t *state_matrices[] = {prepared->constraint_matrix,
                                                          prepared->quadratic_objective_term->objective_sparse_matrix,
                                                          prepared->quadratic_objective_term->objective_lowrank_matrix};
    for (int matrix = 0; matrix < 3; ++matrix)
    {
        const CsrComponent *raw = raw_matrices[matrix];
        const CsrComponent *converted = converted_matrices[matrix];
        const device_sparse_matrix_csr_t *resident = state_matrices[matrix];
        if (converted == raw || converted->row_ptr == raw->row_ptr || converted->col_ind == raw->col_ind ||
            converted->val == raw->val || resident->row_ptr == converted->row_ptr ||
            resident->col_ind == converted->col_ind || resident->val == converted->val)
        {
            fprintf(stderr, "%s: matrix %d aliases its source\n", case_name, matrix);
            ++failures;
        }
        const CsrComponent *before = original_matrices[matrix];
        check_array("unchanged source matrix", raw->val, before->val, resident->num_nonzeros, 0);
        if (memcmp(raw->row_ptr, before->row_ptr, ((size_t)resident->num_rows + 1) * sizeof(int)) ||
            memcmp(raw->col_ind, before->col_ind, (size_t)resident->num_nonzeros * sizeof(int)))
        {
            fprintf(stderr, "%s: source matrix %d structure changed\n", case_name, matrix);
            ++failures;
        }
        device_sparse_matrix_csr_t actual_csr = {
            .row_ptr = converted->row_ptr, .col_ind = converted->col_ind, .val = converted->val};
        device_sparse_matrix_csr_t original_csr = {
            .row_ptr = before->row_ptr, .col_ind = before->col_ind, .val = before->val};
        check_csr(
            "unchanged converted matrix", &actual_csr, &original_csr, resident->num_rows, resident->num_nonzeros, 1);
    }
    check_array("unchanged source D",
                source->objective_lowrank_middle_matrix->val,
                original->objective_lowrank_middle_matrix->val,
                original->objective_lowrank_middle_matrix_num_nonzeros,
                0);
    check_array("unchanged converted D",
                device_prepared->objective_lowrank_middle_matrix->val,
                original->objective_lowrank_middle_matrix->val,
                original->objective_lowrank_middle_matrix_num_nonzeros,
                1);
    double first_scaled;
    copy_to_host(&first_scaled, prepared->constraint_matrix->val, sizeof(first_scaled), 1);
    if (first_scaled == original->constraint_matrix->val[0])
    {
        fprintf(stderr, "%s: state problem was not scaled\n", case_name);
        ++failures;
    }
    free_device_problem(device_prepared);
    const device_sparse_matrix_csr_t *matrices[] = {reference->constraint_matrix,
                                                    reference->quadratic_objective_term->objective_sparse_matrix,
                                                    reference->quadratic_objective_term->objective_lowrank_matrix};
    const int nonzeros[] = {matrices[0]->num_nonzeros, matrices[1]->num_nonzeros, matrices[2]->num_nonzeros};
    double *values[3];
    for (int matrix = 0; matrix < 3; ++matrix)
    {
        values[matrix] = allocate((size_t)nonzeros[matrix], sizeof(double));
        memcpy(values[matrix], matrices[matrix]->val, (size_t)nonzeros[matrix] * sizeof(double));
    }
    double objective[5], columns[5], rows[3], primal[5], dual[3];
    memcpy(objective, reference->objective_vector, sizeof(objective));
    memcpy(columns, reference->variable_rescaling, sizeof(columns));
    memcpy(rows, reference->constraint_rescaling, sizeof(rows));
    for (int j = 0; j < 5; ++j)
        primal[j] = source->primal_start[j] * columns[j] * reference->constraint_bound_rescaling;
    for (int j = 0; j < 3; ++j)
        dual[j] = source->dual_start[j] * rows[j] * reference->objective_vector_rescaling;
    qp_problem_free(source);
    qp_problem_free(original);
    pdhcg_cpu_pdhg_solver_state_free(reference);
    check_array("owned A", prepared->constraint_matrix->val, values[0], nonzeros[0], 1);
    check_array("owned Q", prepared->quadratic_objective_term->objective_sparse_matrix->val, values[1], nonzeros[1], 1);
    check_array(
        "owned R", prepared->quadratic_objective_term->objective_lowrank_matrix->val, values[2], nonzeros[2], 1);
    check_array("owned objective", prepared->objective_vector, objective, 5, 1);
    check_array("owned column scaling", prepared->variable_rescaling, columns, 5, 1);
    check_array("owned row scaling", prepared->constraint_rescaling, rows, 3, 1);
    check_array("owned primal start", prepared->initial_primal_solution, primal, 5, 1);
    check_array("owned dual start", prepared->initial_dual_solution, dual, 3, 1);
    for (int matrix = 0; matrix < 3; ++matrix)
        free(values[matrix]);
    pdhg_solver_state_free(prepared);
}

int main(void)
{
    if (strcmp(pdhcg_device_name(), PDHCG_TEST_DEVICE) != 0)
    {
        fprintf(stderr, "Expected %s backend, got %s\n", PDHCG_TEST_DEVICE, pdhcg_device_name());
        return 1;
    }
    if (pdhcg_device_initialize())
        return 1;
    for (int kind = 0; kind <= 6; ++kind)
    {
        qp_problem_t *problem = make_quadratic_problem(kind);
        if (problem && kind == 6)
        {
            /* Preserve the dense D values while testing duplicate CSR entries. */
            CsrComponent *middle = problem->objective_lowrank_middle_matrix;
            int nnz = problem->objective_lowrank_middle_matrix_num_nonzeros;
            middle->col_ind = realloc(middle->col_ind, ((size_t)nnz + 1) * sizeof(int));
            middle->val = realloc(middle->val, ((size_t)nnz + 1) * sizeof(double));
            if (!middle->col_ind || !middle->val)
                abort();
            middle->col_ind[nnz] = middle->col_ind[nnz - 1];
            middle->val[nnz] = middle->val[nnz - 1];
            ++middle->row_ptr[problem->num_rank_lowrank_obj];
            ++problem->objective_lowrank_middle_matrix_num_nonzeros;
        }
        for (int linearized = 0; linearized <= 1; ++linearized)
        {
            for (int method = 0; method <= 9; ++method)
            {
                snprintf(case_name, sizeof(case_name), "Q=%d linearized=%d method=%d", kind, linearized, method);
                pdhg_parameters_t params = scaling_parameters(method);
                params.non_diagonal_quadratic_mode =
                    linearized ? NON_DIAGONAL_QUADRATIC_LINEARIZED : NON_DIAGONAL_QUADRATIC_INNER;
                compare_scaling(problem, &params);
            }
        }
        qp_problem_free(problem);
    }
    qp_problem_t *cones = make_cone_problem();
    for (int preserving = 0; preserving <= 1; ++preserving)
    {
        for (int method = 0; method <= 9; ++method)
        {
            snprintf(case_name, sizeof(case_name), "all cones preserving=%d method=%d", preserving, method);
            pdhg_parameters_t params = scaling_parameters(method);
            params.use_cone_preserving_scaling = preserving != 0;
            compare_scaling(cones, &params);
        }
    }
    qp_problem_free(cones);
    qp_problem_t *long_row = make_long_row_problem();
    for (int method = 0; method <= 9; ++method)
    {
        snprintf(case_name, sizeof(case_name), "long row with duplicates method=%d", method);
        pdhg_parameters_t params = scaling_parameters(method);
        compare_scaling(long_row, &params);
    }
    qp_problem_free(long_row);
    test_empty_matrix();
    test_unconstrained_quadratic();
    test_large_cone(0, 0);
    test_large_cone(0, 1);
    test_large_cone(1, 0);
    test_large_cone(1, 1);
    test_initialization(0);
    test_initialization(1);
    test_initialization(6);
    test_initialization(-1);
    test_deferred_quadratic();
    test_diagonal_regularization();
    test_preparation_ownership();
    printf("%s device scaling: %d cases, %d failures\n", pdhcg_device_name(), cases, failures);
    return failures ? 1 : 0;
}
