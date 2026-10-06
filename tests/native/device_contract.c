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

#include "device_general_op.h"
#include "device_kernels.h"
#include "pdhcg_psd_cone.h"
#include "spmv_backend.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

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
    psd_projection();
    check(pdhcg_device_synchronize());
    printf("Device contract passed on %s\n", pdhcg_device_name());
    return 0;
}
