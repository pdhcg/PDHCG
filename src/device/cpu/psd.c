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

#include "pdhcg_psd_cone.h"
#include "threads.h"
#include "utils.h"
#include <math.h>
#include <omp.h>
#include <stdint.h>

/* LAPACK's LP64 symmetric eigensolver. Each cone owns its reusable workspace. */
extern void
dsyev_(const char *, const char *, const int *, double *, const int *, double *, double *, const int *, int *);

static void symmetric_eigensolve(int n, double *matrix, double *eigenvalues, double *work, int lwork, int *info)
{
    /* Old OpenMP OpenBLAS derives its global count from the caller's ICV.
     * Scope that ICV as well as the provider setting, preserving the worker's
     * original configuration. MKL's override must be set on this worker. */
#pragma omp parallel num_threads(1)
    {
        omp_set_num_threads(1);
        const int previous = pdhcg_cpu_blas_enter();
        dsyev_("V", "L", &n, matrix, &n, eigenvalues, work, &lwork, info);
        pdhcg_cpu_blas_leave(previous);
    }
}

typedef struct
{
    int start, order, length, lwork;
    double *matrix, *eigenvalues, *work;
} cpu_psd_block_t;
struct psd_projection_runtime_s
{
    int complementarity_offset, count;
    cpu_psd_block_t *blocks;
};
psd_projection_runtime_t *create_psd_projection_runtime(const int *start, const int *order, int count, int offset)
{
    if (count <= 0)
        return NULL;
    psd_projection_runtime_t *runtime = safe_calloc(1, sizeof(*runtime));
    runtime->complementarity_offset = offset;
    runtime->count = count;
    runtime->blocks = safe_calloc((size_t)count, sizeof(*runtime->blocks));
    for (int k = 0; k < count; ++k)
    {
        cpu_psd_block_t *block = &runtime->blocks[k];
        int n = order[k];
        block->start = start[k];
        block->order = n;
        block->length = (int)((int64_t)n * (n + 1) / 2);
        if (n > 1)
        {
            block->matrix = safe_calloc((size_t)n * n, sizeof(double));
            block->eigenvalues = safe_calloc((size_t)n, sizeof(double));
            double query = 0;
            int lwork = -1, info = 0;
            symmetric_eigensolve(n, block->matrix, block->eigenvalues, &query, lwork, &info);
            if (info || !isfinite(query) || query < 1 || query > INT32_MAX)
            {
                fprintf(stderr, "CPU PSD workspace query failed (info=%d)\n", info);
                abort();
            }
            block->lwork = (int)query;
            block->work = safe_malloc((size_t)block->lwork * sizeof(double));
        }
    }
    return runtime;
}
void free_psd_projection_runtime(psd_projection_runtime_t *runtime)
{
    if (!runtime)
        return;
    for (int k = 0; k < runtime->count; ++k)
    {
        free(runtime->blocks[k].matrix);
        free(runtime->blocks[k].eigenvalues);
        free(runtime->blocks[k].work);
    }
    free(runtime->blocks);
    free(runtime);
}
void project_psd_cones(psd_projection_runtime_t *runtime, double *vector)
{
    if (!runtime)
        return;
    int count = runtime->count;
#pragma omp parallel for schedule(static) if (count >= 4)
    for (int k = 0; k < count; ++k)
    {
        cpu_psd_block_t *b = &runtime->blocks[k];
        int n = b->order;
        if (n == 1)
        {
            vector[b->start] = fmax(vector[b->start], 0.0);
            continue;
        }
        int slot = 0;
        bool finite = true;
        for (int col = 0; col < n; ++col)
            for (int row = col; row < n; ++row)
            {
                double v = vector[b->start + slot++];
                finite = finite && isfinite(v);
                if (row != col)
                    v *= 0.70710678118654752440;
                b->matrix[row + col * n] = b->matrix[col + row * n] = v;
            }
        int info = 1;
        if (finite)
            symmetric_eigensolve(n, b->matrix, b->eigenvalues, b->work, b->lwork, &info);
        slot = 0;
        for (int col = 0; col < n; ++col)
            for (int row = col; row < n; ++row)
            {
                double value = 0;
                if (info == 0)
                    for (int j = 0; j < n; ++j)
                        value += fmax(b->eigenvalues[j], 0.0) * b->matrix[row + j * n] * b->matrix[col + j * n];
                else
                    value = NAN;
                if (row != col)
                    value *= 1.41421356237309504880;
                vector[b->start + slot++] = value;
            }
    }
}
void compute_psd_cone_dual_residual(psd_projection_runtime_t *runtime,
                                    double *residual,
                                    const double *objective,
                                    const double *dual,
                                    const double *rescaling)
{
    if (!runtime)
        return;
    for (int k = 0; k < runtime->count; ++k)
    {
        const cpu_psd_block_t *b = &runtime->blocks[k];
        for (int i = b->start; i < b->start + b->length; ++i)
            residual[i] = objective[i] - dual[i];
    }
    project_psd_cones(runtime, residual);
    for (int k = 0; k < runtime->count; ++k)
    {
        const cpu_psd_block_t *b = &runtime->blocks[k];
        for (int i = b->start; i < b->start + b->length; ++i)
            residual[i] = (objective[i] - dual[i] - residual[i]) * rescaling[i];
    }
}
void recompute_psd_cone_reflection(psd_projection_runtime_t *runtime,
                                   double *ref,
                                   const double *primal,
                                   const double *current)
{
    if (!runtime)
        return;
    for (int k = 0; k < runtime->count; ++k)
    {
        const cpu_psd_block_t *b = &runtime->blocks[k];
        for (int i = b->start; i < b->start + b->length; ++i)
            ref[i] = 2 * primal[i] - current[i];
    }
}
void set_psd_cone_dual_slack(psd_projection_runtime_t *runtime,
                             double *slack,
                             const double *objective,
                             const double *dual)
{
    if (!runtime)
        return;
    for (int k = 0; k < runtime->count; ++k)
    {
        const cpu_psd_block_t *b = &runtime->blocks[k];
        for (int i = b->start; i < b->start + b->length; ++i)
            slack[i] = objective[i] - dual[i];
    }
}
void prepare_psd_affine_cone_residuals(psd_projection_runtime_t *runtime,
                                       double *point,
                                       double *complementarity,
                                       const double *product,
                                       const double *offset,
                                       const double *dual,
                                       double scale)
{
    if (!runtime)
        return;
    for (int k = 0; k < runtime->count; ++k)
    {
        const cpu_psd_block_t *b = &runtime->blocks[k];
        double dot = 0;
        for (int i = b->start; i < b->start + b->length; ++i)
        {
            point[i] = -dual[i];
            dot += dual[i] * (product[i] + offset[i]);
        }
        complementarity[runtime->complementarity_offset + k] = fabs(dot) / scale;
    }
}
