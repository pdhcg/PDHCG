/*
Copyright 2025 Haihao Lu
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

#include "device_cones.h"
#include "device_general_op.h"
#include "device_kernels.h"
#include "pdhcg_psd_cone.h"
#include "preprocess.h"
#include "spmv_backend.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void check(int status)
{
    if (status)
    {
        fprintf(stderr, "device status %d\n", status);
        abort();
    }
}
static void near(double actual, double expected)
{
    if (!isfinite(actual) || fabs(actual - expected) > 1e-11 * (1 + fabs(expected)))
    {
        fprintf(stderr, "expected %.17g, got %.17g\n", expected, actual);
        abort();
    }
}
static void *device_buffer(const void *host, size_t bytes)
{
    void *data = NULL;
    check(pdhcg_device_allocate(&data, bytes));
    if (host)
        check(pdhcg_device_copy(data, host, bytes, PDHCG_COPY_HOST_TO_DEVICE));
    else
        check(pdhcg_device_zero(data, 0, bytes));
    return data;
}
static void download(void *host, const void *device, size_t bytes)
{
    check(pdhcg_device_copy(host, device, bytes, PDHCG_COPY_DEVICE_TO_HOST));
}
static double *filled_buffer(int n, double value)
{
    double *host = malloc((size_t)n * sizeof(*host));
    if (!host)
        abort();
    for (int i = 0; i < n; ++i)
        host[i] = value;
    double *data = device_buffer(host, (size_t)n * sizeof(*host));
    free(host);
    return data;
}
static void check_filled(const double *data, int n, double expected)
{
    double *host = malloc((size_t)n * sizeof(*host));
    if (!host)
        abort();
    download(host, data, (size_t)n * sizeof(*host));
    for (int i = 0; i < n; ++i)
        near(host[i], expected);
    free(host);
}
static void reductions(void)
{
    enum
    {
        N = 4099
    };
    pdhcg_device_blas_t blas;
    check(pdhcg_device_blas_create(&blas));
    double x[N], y[N];
    long double reference = 0;
    for (int i = 0; i < N; ++i)
    {
        x[i] = i % 19 - 9;
        y[i] = (i % 11 - 5) * 0.25;
        reference += (long double)x[i] * y[i];
    }
    double *dx = device_buffer(x, sizeof(x));
    double *dy = device_buffer(y, sizeof(y));
    double *scalar = filled_buffer(1, NAN);
    double result = 0;
    check(pdhcg_device_dot(blas, N, dx, 1, dy, 1, &result));
    near(result, (double)reference);
    check(pdhcg_device_set_pointer_mode(blas, PDHCG_POINTER_DEVICE));
    check(pdhcg_device_dot(blas, N, dx, 1, dy, 1, scalar));
    download(&result, scalar, sizeof(result));
    near(result, (double)reference);
    check(pdhcg_device_set_pointer_mode(blas, PDHCG_POINTER_HOST));
    const double huge[] = {3e200, 4e200};
    double *dh = device_buffer(huge, sizeof(huge));
    check(pdhcg_device_nrm2(blas, 2, dh, 1, &result));
    near(result, 5e200);
    check(pdhcg_device_dot(blas, 0, dx, 1, dy, 1, &result));
    near(result, 0);
    check(pdhcg_device_free(dx));
    check(pdhcg_device_free(dy));
    check(pdhcg_device_free(dh));
    check(pdhcg_device_free(scalar));
    check(pdhcg_device_blas_destroy(blas));
}
static void sparse_operations(void)
{
    enum
    {
        M = 2053,
        N = 31,
        NNZ = M * 2
    };
    pdhcg_device_sparse_t sparse;
    check(pdhcg_device_sparse_create(&sparse));
    int rows[M + 1], columns[NNZ];
    double values[NNZ], x[N], reference[M], got[M], expected[N] = {0};
    for (int j = 0; j < N; ++j)
        x[j] = j - 7;
    for (int i = 0; i < M; ++i)
    {
        rows[i] = i * 2;
        columns[2 * i] = i % N;
        columns[2 * i + 1] = (i + 7) % N;
        values[2 * i] = 2;
        values[2 * i + 1] = -0.5;
        reference[i] = 2 * x[i % N] - 0.5 * x[(i + 7) % N];
    }
    rows[M] = NNZ;
    int *dr = device_buffer(rows, sizeof(rows));
    int *dc = device_buffer(columns, sizeof(columns));
    int *tr = device_buffer(NULL, (N + 1) * sizeof(int));
    int *tc = device_buffer(NULL, sizeof(columns));
    double *dv = device_buffer(values, sizeof(values));
    double *dx = device_buffer(x, sizeof(x));
    double *dy = filled_buffer(M, NAN);
    double *tv = device_buffer(NULL, sizeof(values));
    double *tz = filled_buffer(N, NAN);
    pdhcg_device_vector_t vx, vy, vz;
    check(pdhcg_device_vector_create(&vx, N, dx));
    check(pdhcg_device_vector_create(&vy, M, dy));
    check(pdhcg_device_vector_create(&vz, N, tz));
    pdhcg_spmv_ctx_t *a = pdhcg_spmv_ctx_create(sparse, M, N, NNZ, dr, dc, dv, vx, vy);
    const double one = 1, zero = 0;
    pdhcg_spmv_execute(sparse, a, &one, &zero, dx, dy);
    download(got, dy, sizeof(got));
    for (int i = 0; i < M; ++i)
        near(got[i], reference[i]);
    check(pdhcg_device_csr_transpose(sparse, M, N, NNZ, dv, dr, dc, tv, tr, tc));
    pdhcg_spmv_ctx_t *at = pdhcg_spmv_ctx_create(sparse, N, M, NNZ, tr, tc, tv, vy, vz);
    pdhcg_spmv_execute(sparse, at, &one, &zero, dy, tz);
    for (int i = 0; i < M; ++i)
        for (int k = rows[i]; k < rows[i + 1]; ++k)
            expected[columns[k]] += values[k] * reference[i];
    download(got, tz, sizeof(expected));
    for (int j = 0; j < N; ++j)
        near(got[j], expected[j]);
    pdhcg_spmv_ctx_destroy(a);
    pdhcg_spmv_ctx_destroy(at);
    /* Empty CSR input only needs the caller's output row array. */
    check(pdhcg_device_csr_transpose(sparse, M, N, 0, NULL, NULL, NULL, NULL, tr, NULL));
    int empty_rows[N + 1];
    download(empty_rows, tr, sizeof(empty_rows));
    for (int j = 0; j <= N; ++j)
        if (empty_rows[j] != 0)
            abort();
    const int sentinel = -1;
    check(pdhcg_device_copy(tr, &sentinel, sizeof(sentinel), PDHCG_COPY_HOST_TO_DEVICE));
    check(pdhcg_device_csr_transpose(sparse, 0, 0, 0, NULL, NULL, NULL, NULL, tr, NULL));
    download(empty_rows, tr, sizeof(int));
    if (empty_rows[0] != 0)
        abort();
    check(pdhcg_device_vector_destroy(vx));
    check(pdhcg_device_vector_destroy(vy));
    check(pdhcg_device_vector_destroy(vz));
    check(pdhcg_device_sparse_destroy(sparse));
    void *buffers[] = {dr, dc, tr, tc, dv, dx, dy, tv, tz};
    for (size_t i = 0; i < sizeof(buffers) / sizeof(buffers[0]); ++i)
        check(pdhcg_device_free(buffers[i]));
}
static void fused_primal_dual_update(void)
{
    /* Unequal dimensions exercise both halves of the concatenated range. */
    const int n = 2049, m = 3007;
    double *ax = filled_buffer(n, 1), *x = filled_buffer(n, 2), *rx = filled_buffer(n, 3);
    double *ay = filled_buffer(m, 4), *y = filled_buffer(m, 5), *ry = filled_buffer(m, 6);
    pdhcg_device_halpern_update(ax, x, rx, ay, y, ry, n, m, 0.75, 0.5);
    check_filled(x, n, 2.125);
    check_filled(y, m, 5.125);
    pdhcg_device_compute_delta_solution(ax, x, rx, ay, y, ry, n, m);
    check_filled(rx, n, 1.125);
    check_filled(ry, m, 1.125);
    void *buffers[] = {ax, x, rx, ay, y, ry};
    for (size_t i = 0; i < sizeof(buffers) / sizeof(buffers[0]); ++i)
        check(pdhcg_device_free(buffers[i]));
}
static void preprocessing(void)
{
    /* A column shard keeps global CSR rows and local column indices. Upload
       all five rows before exposing the three local variables to preprocessing. */
    int rows[] = {0, 0, 0, 2, 2, 3}, columns[] = {0, 0, 2};
    double values[] = {10.0, 1e-12, 0.0};
    CsrComponent q = {.row_ptr = rows, .col_ind = columns, .val = values};
    qp_problem_t source = {0};
    source.num_variables = 5;
    source.objective_sparse_matrix = &q;
    source.objective_sparse_matrix_num_nonzeros = 3;
    qp_problem_t *device = convert_problem_to_device(&source, NULL);
    source.num_variables = device->num_variables = 3;
    const double scaled_value = 4e-12;
    check(pdhcg_device_copy(
        device->objective_sparse_matrix->val + 1, &scaled_value, sizeof(scaled_value), PDHCG_COPY_HOST_TO_DEVICE));
    processed_qp_problem_t *processed = preprocess_qp_problem(&source, device, PDHCG_DIAG_Q, 2);
    double got[9];
    download(got, processed->diagonal_quad_objective, 3 * sizeof(double));
    /* Preprocessing preserves raw values; regularization follows scaling. */
    if (got[0] != scaled_value || got[1] != 0.0 || got[2] != 0.0 ||
        processed->objective_sparse_matrix != device->objective_sparse_matrix || values[1] != 1e-12)
        abort();
    free_processed_qp_problem(processed);
    download(got, device->objective_sparse_matrix->val, sizeof(values));
    if (got[0] != 10.0 || got[1] != scaled_value || got[2] != 0.0)
        abort();
    free_device_problem(device);

    for (int dense = 0; dense < 2; ++dense)
    {
        int middle_rows[] = {0, 3, 3, 4};
        int middle_columns[] = {0, dense ? 2 : 0, dense ? 2 : 0, 2};
        double middle_values[] = {2, 3, 7, 5};
        CsrComponent middle = {.row_ptr = middle_rows, .col_ind = middle_columns, .val = middle_values};
        int rank_rows[] = {0, 1, 2, 3}, rank_columns[] = {0, 1, 2};
        double rank_values[] = {1, 1, 1};
        CsrComponent rank = {.row_ptr = rank_rows, .col_ind = rank_columns, .val = rank_values};
        qp_problem_t lowrank = {0};
        lowrank.num_variables = lowrank.num_rank_lowrank_obj = 3;
        lowrank.objective_lowrank_matrix = &rank;
        lowrank.objective_lowrank_matrix_num_nonzeros = 3;
        lowrank.objective_lowrank_middle_matrix = &middle;
        lowrank.objective_lowrank_middle_matrix_num_nonzeros = 4;
        device = convert_problem_to_device(&lowrank, NULL);
        processed = preprocess_qp_problem(&lowrank, device, PDHCG_LOW_RANK_Q, 0);
        if (processed->objective_lowrank_middle_kind != (dense ? PDHCG_D_DENSE : PDHCG_D_DIAG) ||
            processed->objective_lowrank_matrix != device->objective_lowrank_matrix)
            abort();
        const double expected_diag[] = {7, 0, 5};
        const double expected_dense[] = {2, 0, 7, 0, 0, 0, 0, 0, 5};
        int count = dense ? 9 : 3;
        download(got,
                 dense ? processed->objective_lowrank_middle_dense : processed->objective_lowrank_middle_diag,
                 (size_t)count * sizeof(double));
        for (int i = 0; i < count; ++i)
            near(got[i], dense ? expected_dense[i] : expected_diag[i]);
        free_processed_qp_problem(processed);
        download(got, device->objective_lowrank_middle_matrix->val, sizeof(middle_values));
        const double original_middle[] = {2, 3, 7, 5};
        for (int i = 0; i < 4; ++i)
        {
            near(got[i], original_middle[i]);
            near(middle_values[i], original_middle[i]);
        }
        free_device_problem(device);
    }
}
static void cone_methods(void)
{
    const int large = PDHCG_LARGE_CONE_MIN_VDIM;
    const struct
    {
        cone_type_t type;
        int dimension;
        cone_proj_method_t method;
        cone_proj_method_t without_scaling;
    } cases[] = {
        {CONE_STANDARD_SOC, 15, PROJ_METHOD_THREAD, PROJ_METHOD_THREAD},
        {CONE_STANDARD_SOC, 64, PROJ_METHOD_BLOCK, PROJ_METHOD_BLOCK},
        {CONE_ROTATED_SOC, 64, PROJ_METHOD_WARP, PROJ_METHOD_WARP},
        {CONE_STANDARD_SOC, large, PROJ_METHOD_GRID, PROJ_METHOD_WARP},
        {CONE_STANDARD_SOC, large, PROJ_METHOD_GRID_WEIGHTED, PROJ_METHOD_WARP},
        {CONE_STANDARD_SOC, large, PROJ_METHOD_GRID_WEIGHTED, PROJ_METHOD_GRID_WEIGHTED},
        {CONE_ROTATED_SOC, large, PROJ_METHOD_GRID, PROJ_METHOD_WARP},
        {CONE_ROTATED_SOC, large, PROJ_METHOD_GRID, PROJ_METHOD_WARP},
        {CONE_ROTATED_SOC, large, PROJ_METHOD_GRID_WEIGHTED, PROJ_METHOD_WARP},
        {CONE_ROTATED_SOC, large, PROJ_METHOD_GRID_WEIGHTED, PROJ_METHOD_GRID_WEIGHTED},
        {CONE_STANDARD_SOC, large, PROJ_METHOD_WARP, PROJ_METHOD_WARP},
        {CONE_POWER, 1, PROJ_METHOD_THREAD, PROJ_METHOD_THREAD},
        {CONE_STANDARD_SOC, large, PROJ_METHOD_GRID_WEIGHTED, PROJ_METHOD_WARP},
    };
    enum
    {
        COUNT = sizeof(cases) / sizeof(cases[0])
    };
    int starts[COUNT], dimensions[COUNT], length = 3;
    cone_type_t types[COUNT];
    for (int i = 0; i < COUNT; ++i)
    {
        starts[i] = length;
        dimensions[i] = cases[i].dimension;
        types[i] = cases[i].type;
        length += dimensions[i] + 2;
    }
    double *scaling = malloc((size_t)length * sizeof(double));
    char *fixed = calloc((size_t)length, sizeof(char));
    if (!scaling || !fixed)
        abort();
    for (int i = 0; i < length; ++i)
        scaling[i] = 4.0;
    fixed[starts[1] + 3] = fixed[starts[5] + large - 1] = fixed[starts[9] + large + 1] = 1;
    scaling[starts[4] + large - 1] = 8.0;
    /* Scalar-uniform RSOC must not become weighted due to sqrt rounding. */
    for (int i = 0; i < large + 2; ++i)
        scaling[starts[6] + i] = 0.7;
    for (int i = 0; i < large; ++i)
        scaling[starts[7] + i] = 2.0;
    scaling[starts[7] + large] = scaling[starts[8] + large] = 1.0;
    scaling[starts[10] + large + 1] = NAN;
    scaling[starts[12] + large] = 2.0;
    cone_blocks_t cones = {.num_cones = COUNT,
                           .start_idx = starts,
                           .v_dim = dimensions,
                           .type = types,
                           .fixed_mask_size = length,
                           .is_fixed = fixed};
    double *device_scaling = device_buffer(scaling, (size_t)length * sizeof(double));
    int cpu = strcmp(pdhcg_device_name(), "cpu") == 0;
    /* Reuse the same data for non-NULL scaling with zero or one candidate. */
    const int begin[] = {0, 0, 0, COUNT - 1}, count[] = {COUNT, COUNT, 3, 1};
    for (int batch = 0; batch < 4; ++batch)
    {
        cone_blocks_t subset = cones;
        subset.num_cones = count[batch];
        subset.start_idx += begin[batch];
        subset.v_dim += begin[batch];
        subset.type += begin[batch];
        cone_proj_method_t methods[COUNT];
        for (int i = 0; i < COUNT; ++i)
            methods[i] = NUM_PROJ_METHODS;
        pdhcg_device_cone_methods(&subset, batch ? device_scaling : NULL, methods);
        for (int i = 0; i < subset.num_cones; ++i)
        {
            int source = begin[batch] + i;
            cone_proj_method_t want = cpu ? PROJ_METHOD_THREAD
                : batch                   ? cases[source].method
                                          : cases[source].without_scaling;
            if (methods[i] != want)
            {
                fprintf(stderr, "cone method[%d], batch=%d: got %d, expected %d\n", i, batch, methods[i], want);
                abort();
            }
        }
    }
    check(pdhcg_device_free(device_scaling));
    free(scaling);
    free(fixed);
}

static void cone_scaling_and_primal(void)
{
    enum
    {
        N = 11,
        M = 5
    };
    /* Layout is bucket-permuted: RSOC precedes SOC despite its later coordinates. */
    const int starts[] = {4, 0, 8}, dimensions[] = {2, 2, 1};
    cone_bucket_t buckets[] = {{CONE_ROTATED_SOC, PROJ_METHOD_THREAD, 0, 1},
                               {CONE_STANDARD_SOC, PROJ_METHOD_THREAD, 1, 1},
                               {CONE_EXPONENTIAL, PROJ_METHOD_THREAD, 2, 1}};
    const double row_scaling[] = {0.25, 0.5, 1, 2, 8};
    double scaling[N], initial[N], expected[N], got[N];
    for (int i = 0; i < N; ++i)
        scaling[i] = 0.5 * (i + 1);
    pdhg_solver_state_t state = {0};
    state.num_variables = N;
    state.num_constraints = M;
    state.constraint_bound_rescaling = 0.25;
    state.has_variable_cones = true;
    state.variable_rescaling = device_buffer(scaling, sizeof(scaling));
    state.constraint_rescaling = device_buffer(row_scaling, sizeof(row_scaling));
    state.initial_primal_solution = filled_buffer(N, 0);
    state.current_primal_solution = filled_buffer(N, 19);
    state.pdhg_primal_solution = filled_buffer(N, 19);
    state.reflected_primal_solution = filled_buffer(N, 19);
    state.cones.num_blocks = state.cones.num_buckets = 3;
    state.cones.buckets = buckets;
    state.cones.start_idx = device_buffer(starts, sizeof(starts));
    state.cones.v_dim = device_buffer(dimensions, sizeof(dimensions));
    state.affine_cones.axis = CONE_AXIS_CONSTRAINT;
    state.affine_cones.num_blocks = 1;
    state.affine_cones.coordinate_rescaling = filled_buffer(M, NAN);
    for (int scenario = 0; scenario < 3; ++scenario)
    {
        char fixed[N] = {0};
        if (scenario == 1)
            fixed[2] = fixed[7] = 1;
        if (scenario == 2)
            fixed[3] = 1;
        for (int i = 0; i < N; ++i)
            initial[i] = expected[i] = scenario ? 10 + i : 0;
        if (!fixed[2])
            expected[2] = -0.7071067811865475 * 0.25 * scaling[2];
        if (!fixed[3])
            expected[3] = 0.7071067811865475 * 0.25 * scaling[3];
        if (!fixed[7])
            expected[7] = 0.25 * scaling[7];
        state.cones.is_fixed = scenario ? device_buffer(fixed, sizeof(fixed)) : NULL;
        check(pdhcg_device_copy(state.initial_primal_solution, initial, sizeof(initial), PDHCG_COPY_HOST_TO_DEVICE));
        pdhcg_device_initialize_cone_scaling_and_primal(&state);
        download(got, state.initial_primal_solution, sizeof(got));
        for (int i = 0; i < N; ++i)
            near(got[i], expected[i]);
        download(got, state.affine_cones.coordinate_rescaling, sizeof(row_scaling));
        for (int i = 0; i < M; ++i)
            near(got[i], 1.0 / row_scaling[i]);
        check_filled(state.current_primal_solution, N, 19);
        check_filled(state.pdhg_primal_solution, N, 19);
        check_filled(state.reflected_primal_solution, N, 19);
        check(pdhcg_device_free(state.cones.is_fixed));
    }
    void *buffers[] = {state.variable_rescaling,
                       state.constraint_rescaling,
                       state.initial_primal_solution,
                       state.current_primal_solution,
                       state.pdhg_primal_solution,
                       state.reflected_primal_solution,
                       state.cones.start_idx,
                       state.cones.v_dim,
                       state.affine_cones.coordinate_rescaling};
    for (size_t i = 0; i < sizeof(buffers) / sizeof(buffers[0]); ++i)
        check(pdhcg_device_free(buffers[i]));
}

static void psd_projection(void)
{
    int starts[9], orders[9];
    double point[27] = {0}, expected[27], got[27];
    for (int k = 0; k < 9; ++k)
    {
        starts[k] = k * 3;
        orders[k] = 2;
        point[k * 3 + 1] = 2 * sqrt(2.0);
        expected[k * 3] = expected[k * 3 + 2] = 1;
        expected[k * 3 + 1] = sqrt(2.0);
    }
    double *v = device_buffer(point, sizeof(point));
    psd_projection_runtime_t *runtime = create_psd_projection_runtime(starts, orders, 9, 0);
    project_psd_cones(runtime, v);
    download(got, v, sizeof(got));
    for (int i = 0; i < 27; ++i)
        near(got[i], expected[i]);
    project_psd_cones(runtime, v);
    download(got, v, sizeof(got));
    for (int i = 0; i < 27; ++i)
        near(got[i], expected[i]);
    free_psd_projection_runtime(runtime);
    check(pdhcg_device_free(v));
}
int main(void)
{
    check(pdhcg_device_initialize());
    reductions();
    sparse_operations();
    fused_primal_dual_update();
    preprocessing();
    cone_methods();
    cone_scaling_and_primal();
    psd_projection();
    check(pdhcg_device_synchronize());
    printf("Device contract passed on %s\n", pdhcg_device_name());
    return 0;
}
