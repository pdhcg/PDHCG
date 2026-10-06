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
#include "spmv_backend.h"
#include "utils.h"
#include <math.h>
#include <omp.h>
#include <stdlib.h>
#include <string.h>

struct pdhcg_device_blas
{
    pdhcg_device_pointer_mode_t mode;
};
struct pdhcg_device_sparse
{
    int reserved;
};
struct pdhcg_device_vector
{
    int64_t size;
    double *values;
};
struct pdhcg_spmv_ctx
{
    int m;
    const int *rows;
    const int *columns;
    const double *values;
};

const char *pdhcg_device_name(void)
{
    return "cpu";
}
int pdhcg_device_initialize(void)
{
    return 0;
}
int pdhcg_device_allocate(void **ptr, size_t bytes)
{
    *ptr = malloc(bytes ? bytes : 1);
    return *ptr ? 0 : 1;
}
int pdhcg_device_free(void *ptr)
{
    free(ptr);
    return 0;
}
int pdhcg_device_copy(void *dst, const void *src, size_t bytes, pdhcg_device_copy_kind_t kind)
{
    (void)kind;
    if (bytes)
        memmove(dst, src, bytes);
    return 0;
}
int pdhcg_device_copy_async(void *dst, const void *src, size_t bytes, pdhcg_device_copy_kind_t kind)
{
    return pdhcg_device_copy(dst, src, bytes, kind);
}
int pdhcg_device_zero(void *dst, int value, size_t bytes)
{
    if (bytes)
        memset(dst, value, bytes);
    return 0;
}
int pdhcg_device_zero_async(void *dst, int value, size_t bytes)
{
    return pdhcg_device_zero(dst, value, bytes);
}
int pdhcg_device_synchronize(void)
{
    return 0;
}
int pdhcg_device_last_error(void)
{
    return 0;
}
int pdhcg_device_blas_create(pdhcg_device_blas_t *h)
{
    *h = malloc(sizeof(**h));
    if (*h)
        (*h)->mode = PDHCG_POINTER_HOST;
    return *h ? 0 : 1;
}
int pdhcg_device_blas_destroy(pdhcg_device_blas_t h)
{
    free(h);
    return 0;
}
int pdhcg_device_sparse_create(pdhcg_device_sparse_t *h)
{
    *h = calloc(1, sizeof(**h));
    return *h ? 0 : 1;
}
int pdhcg_device_sparse_destroy(pdhcg_device_sparse_t h)
{
    free(h);
    return 0;
}
int pdhcg_device_set_pointer_mode(pdhcg_device_blas_t h, pdhcg_device_pointer_mode_t m)
{
    h->mode = m;
    return 0;
}
int pdhcg_device_get_pointer_mode(pdhcg_device_blas_t h, pdhcg_device_pointer_mode_t *m)
{
    *m = h->mode;
    return 0;
}
int pdhcg_device_dot(pdhcg_device_blas_t handle, int n, const double *x, int sx, const double *y, int sy, double *out)
{
    (void)handle;
    double sum = 0;
#pragma omp parallel for reduction(+ : sum) schedule(static) if (n >= 1024)
    for (int i = 0; i < n; ++i)
        sum += x[i * sx] * y[i * sy];
    *out = sum;
    return 0;
}
int pdhcg_device_nrm2(pdhcg_device_blas_t handle, int64_t n, const double *x, int64_t sx, double *out)
{
    (void)handle;
    // Scale before summing squares, including on CPU, to avoid overflow/underflow.
    double scale = 0;
    int nonfinite = 0;
#pragma omp parallel for reduction(max : scale) reduction(| : nonfinite) schedule(static) if (n >= 1024)
    for (int64_t i = 0; i < n; ++i)
    {
        double a = fabs(x[i * sx]);
        if (!isfinite(a))
            nonfinite |= isnan(a) ? 2 : 1;
        else
            scale = fmax(scale, a);
    }
    if (nonfinite)
    {
        *out = (nonfinite & 2) ? NAN : INFINITY;
        return 0;
    }
    double sum = 0;
    if (scale > 0)
    {
#pragma omp parallel for reduction(+ : sum) schedule(static) if (n >= 1024)
        for (int64_t i = 0; i < n; ++i)
        {
            double a = x[i * sx] / scale;
            sum += a * a;
        }
    }
    *out = scale * sqrt(sum);
    return 0;
}
int pdhcg_device_asum(pdhcg_device_blas_t handle, int n, const double *x, int sx, double *out)
{
    (void)handle;
    double sum = 0;
#pragma omp parallel for reduction(+ : sum) schedule(static) if (n >= 1024)
    for (int i = 0; i < n; ++i)
        sum += fabs(x[i * sx]);
    *out = sum;
    return 0;
}
int pdhcg_device_iamax(pdhcg_device_blas_t handle, int n, const double *x, int sx, int *out)
{
    (void)handle;
    *out = n > 0 ? 1 : 0;
    for (int i = 1; i < n; ++i)
        if (fabs(x[i * sx]) > fabs(x[(*out - 1) * sx]) || isnan(x[i * sx]))
            *out = i + 1;
    return 0;
}
int pdhcg_device_scal(pdhcg_device_blas_t handle, int n, const double *alpha, double *x, int sx)
{
    (void)handle;
    const double a = *alpha;
#pragma omp parallel for schedule(static) if (n >= 1024)
    for (int i = 0; i < n; ++i)
        x[i * sx] *= a;
    return 0;
}
int pdhcg_device_axpy(
    pdhcg_device_blas_t handle, int n, const double *alpha, const double *x, int sx, double *y, int sy)
{
    (void)handle;
    const double a = *alpha;
#pragma omp parallel for schedule(static) if (n >= 1024)
    for (int i = 0; i < n; ++i)
        y[i * sy] += a * x[i * sx];
    return 0;
}
int pdhcg_device_symv(pdhcg_device_blas_t handle,
                      pdhcg_device_triangle_t triangle,
                      int n,
                      const double *alpha,
                      const double *a,
                      int lda,
                      const double *x,
                      int sx,
                      const double *beta,
                      double *y,
                      int sy)
{
    (void)handle;
    const double ca = *alpha, cb = *beta;
#pragma omp parallel for schedule(static) if (n >= 64)
    for (int i = 0; i < n; ++i)
    {
        double sum = 0;
        for (int j = 0; j < n; ++j)
        {
            bool stored = triangle == PDHCG_TRIANGLE_LOWER ? i >= j : i <= j;
            sum += a[stored ? i + j * lda : j + i * lda] * x[j * sx];
        }
        y[i * sy] = ca * sum + (cb == 0 ? 0 : cb * y[i * sy]);
    }
    return 0;
}
int pdhcg_device_vector_create(pdhcg_device_vector_t *v, int64_t n, double *x)
{
    *v = malloc(sizeof(**v));
    if (*v)
    {
        (*v)->size = n;
        (*v)->values = x;
    }
    return *v ? 0 : 1;
}
int pdhcg_device_vector_destroy(pdhcg_device_vector_t v)
{
    free(v);
    return 0;
}
int pdhcg_device_vector_set_values(pdhcg_device_vector_t v, double *x)
{
    v->values = x;
    return 0;
}
int pdhcg_device_csr_transpose(pdhcg_device_sparse_t handle,
                               int m,
                               int n,
                               int nnz,
                               const double *values,
                               const int *rows,
                               const int *columns,
                               double *ov,
                               int *orr,
                               int *oc)
{
    (void)handle;
    memset(orr, 0, ((size_t)n + 1) * sizeof(*orr));
    for (int k = 0; k < nnz; ++k)
        ++orr[columns[k] + 1];
    for (int j = 0; j < n; ++j)
        orr[j + 1] += orr[j];
    int *next = malloc((n > 0 ? (size_t)n : 1) * sizeof(*next));
    if (!next)
        return 1;
    memcpy(next, orr, (size_t)n * sizeof(*next));
    for (int i = 0; i < m; ++i)
        for (int k = rows[i]; k < rows[i + 1]; ++k)
        {
            int p = next[columns[k]]++;
            oc[p] = i;
            ov[p] = values[k];
        }
    free(next);
    return 0;
}
bool pdhcg_use_spmvop_by_default(void)
{
    return false;
}
pdhcg_spmv_ctx_t *pdhcg_spmv_ctx_create(pdhcg_device_sparse_t handle,
                                        int m,
                                        int n,
                                        int nnz,
                                        int *rows,
                                        int *columns,
                                        double *values,
                                        pdhcg_device_vector_t vec_x,
                                        pdhcg_device_vector_t vec_y)
{
    (void)handle;
    (void)n;
    (void)nnz;
    (void)vec_x;
    (void)vec_y;
    pdhcg_spmv_ctx_t *ctx = safe_malloc(sizeof(*ctx));
    ctx->m = m;
    ctx->rows = rows;
    ctx->columns = columns;
    ctx->values = values;
    return ctx;
}
void pdhcg_spmv_ctx_destroy(pdhcg_spmv_ctx_t *ctx)
{
    free(ctx);
}
void pdhcg_spmv_execute(pdhcg_device_sparse_t handle,
                        pdhcg_spmv_ctx_t *ctx,
                        const double *alpha,
                        const double *beta,
                        const double *x,
                        double *y)
{
    (void)handle;
    const double a = *alpha, b = *beta;
#pragma omp parallel for schedule(static) if (ctx->m >= 256)
    for (int i = 0; i < ctx->m; ++i)
    {
        double sum = 0;
        for (int k = ctx->rows[i]; k < ctx->rows[i + 1]; ++k)
            sum += ctx->values[k] * x[ctx->columns[k]];
        y[i] = a * sum + (b == 0 ? 0 : b * y[i]);
    }
}
