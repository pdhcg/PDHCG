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

#include "checks.h"
#include "device_cones.h"
#include "pdhcg_affine_cone_kernels.h"
#include "utils.h"
#include <math.h>

typedef struct
{
    int cone_index;
    int start_idx;
    int v_dim;
    cone_type_t type;
    cone_proj_method_t method;
} cone_method_candidate_t;

__global__ static void classify_large_cones_kernel(cone_method_candidate_t *candidates, const double *rescaling)
{
    cone_method_candidate_t *candidate = candidates + blockIdx.x;
    int start = candidate->start_idx;
    int dimension = candidate->v_dim;
    double d0 = rescaling[start + dimension];
    double d1 = rescaling[start + dimension + 1];
    if (!(d0 > 0.0) || !(d1 > 0.0) || !isfinite(d0) || !isfinite(d1))
    {
        if (threadIdx.x == 0)
            candidate->method = PROJ_METHOD_WARP;
        return;
    }

    double d_vector = d1;
    if (candidate->type == CONE_STANDARD_SOC)
    {
        if (d0 != d1)
        {
            if (threadIdx.x == 0)
                candidate->method = PROJ_METHOD_GRID_WEIGHTED;
            return;
        }
    }
    else
    {
        double d_ref = rescaling[start];
        bool mismatch = d0 != d_ref || d1 != d_ref;
        for (int i = threadIdx.x; i < dimension; i += blockDim.x)
            mismatch |= rescaling[start + i] != d_ref;
        if (!__syncthreads_or(mismatch))
        {
            if (threadIdx.x == 0)
                candidate->method = PROJ_METHOD_GRID;
            return;
        }
        d_vector = sqrt(d0) * sqrt(d1);
    }

    bool mismatch = false;
    for (int i = threadIdx.x; i < dimension; i += blockDim.x)
        mismatch |= rescaling[start + i] != d_vector;
    bool weighted = __syncthreads_or(mismatch);
    if (threadIdx.x == 0)
        candidate->method = weighted ? PROJ_METHOD_GRID_WEIGHTED : PROJ_METHOD_GRID;
}

void pdhcg_device_cone_methods(const cone_blocks_t *cones, const double *rescaling, cone_proj_method_t *methods)
{
    cone_method_candidate_t *candidates = NULL;
    size_t count = 0, capacity = 0;
    for (int cone = 0; cone < cones->num_cones; ++cone)
    {
        cone_type_t type = cones->type[cone];
        int dimension = cones->v_dim[cone];
        methods[cone] = PROJ_METHOD_THREAD;
        if (type == CONE_EXPONENTIAL || type == CONE_POWER || type == CONE_PSD || dimension < 32)
            continue;
        methods[cone] = PROJ_METHOD_WARP;
        if (type != CONE_STANDARD_SOC && type != CONE_ROTATED_SOC)
            continue;

        bool fixed = false;
        if (cones->is_fixed)
            for (int slot = cones->start_idx[cone]; slot < cones->start_idx[cone] + dimension + 2 && !fixed; ++slot)
                fixed = cones->is_fixed[slot] != 0;
        if (fixed)
        {
            methods[cone] = dimension >= PDHCG_LARGE_CONE_MIN_VDIM ? PROJ_METHOD_GRID_WEIGHTED : PROJ_METHOD_BLOCK;
            continue;
        }
        if (dimension >= PDHCG_LARGE_CONE_MIN_VDIM && rescaling)
        {
            if (count == capacity)
            {
                capacity = capacity ? 2 * capacity : 4;
                candidates = (cone_method_candidate_t *)safe_realloc(candidates, capacity * sizeof(*candidates));
            }
            candidates[count++] = {cone, cones->start_idx[cone], dimension, type, PROJ_METHOD_WARP};
        }
    }
    if (!count)
        return;

    size_t bytes = count * sizeof(*candidates);
    cone_method_candidate_t *device_candidates = NULL;
    DEVICE_CHECK(pdhcg_device_allocate((void **)&device_candidates, bytes));
    DEVICE_CHECK(pdhcg_device_copy(device_candidates, candidates, bytes, PDHCG_COPY_HOST_TO_DEVICE));
    classify_large_cones_kernel<<<count, THREADS_PER_BLOCK>>>(device_candidates, rescaling);
    CUDA_CHECK(cudaGetLastError());
    DEVICE_CHECK(pdhcg_device_copy(candidates, device_candidates, bytes, PDHCG_COPY_DEVICE_TO_HOST));
    for (size_t index = 0; index < count; ++index)
        methods[candidates[index].cone_index] = candidates[index].method;
    DEVICE_CHECK(pdhcg_device_free(device_candidates));
    free(candidates);
}

__global__ static void initialize_affine_scaling_kernel(const double *rescaling, double *inverse, int count)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < count)
        inverse[row] = 1.0 / rescaling[row];
}

__global__ static void initialize_variable_cone_primal_kernel(const int *start_idx,
                                                              const int *v_dim,
                                                              const char *is_fixed,
                                                              const double *rescaling,
                                                              double bound_scale,
                                                              double *initial,
                                                              bool standard_soc,
                                                              int count)
{
    int cone = blockIdx.x * blockDim.x + threadIdx.x;
    if (cone >= count)
        return;
    const double INV_SQRT2 = 0.7071067811865475;
    int aux0 = start_idx[cone] + v_dim[cone];
    if (standard_soc)
    {
        if (!is_fixed || !is_fixed[aux0])
            initial[aux0] = -INV_SQRT2 * bound_scale * rescaling[aux0];
        if (!is_fixed || !is_fixed[aux0 + 1])
            initial[aux0 + 1] = INV_SQRT2 * bound_scale * rescaling[aux0 + 1];
    }
    else if (!is_fixed || !is_fixed[aux0 + 1])
        initial[aux0 + 1] = bound_scale * rescaling[aux0 + 1];
}

void pdhcg_device_initialize_cone_scaling_and_primal(pdhg_solver_state_t *state)
{
    if (state->affine_cones.coordinate_rescaling && state->num_constraints > 0)
        initialize_affine_scaling_kernel<<<(state->num_constraints + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK,
                                           THREADS_PER_BLOCK>>>(
            state->constraint_rescaling, state->affine_cones.coordinate_rescaling, state->num_constraints);
    const cone_runtime_t *cones = &state->cones;
    for (int b = 0; b < cones->num_buckets; ++b)
    {
        const cone_bucket_t *bucket = &cones->buckets[b];
        if (bucket->type == CONE_EXPONENTIAL || bucket->type == CONE_POWER || bucket->type == CONE_PSD)
            continue;
        if (bucket->count > 0)
            initialize_variable_cone_primal_kernel<<<(bucket->count + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK,
                                                     THREADS_PER_BLOCK>>>(cones->start_idx + bucket->offset,
                                                                          cones->v_dim + bucket->offset,
                                                                          cones->is_fixed,
                                                                          state->variable_rescaling,
                                                                          state->constraint_bound_rescaling,
                                                                          state->initial_primal_solution,
                                                                          bucket->type == CONE_STANDARD_SOC,
                                                                          bucket->count);
    }
}
void pdhcg_device_prepare_affine_residuals(pdhg_solver_state_t *state, double *projection_point)
{
    int threads = THREADS_PER_BLOCK;
    for (int bucket_idx = 0; bucket_idx < state->affine_cones.num_buckets; ++bucket_idx)
    {
        const cone_bucket_t *bucket = &state->affine_cones.buckets[bucket_idx];
        double *complementarity = state->affine_cones.complementarity_residual + bucket->offset;
        const int *start_idx = state->affine_cones.start_idx + bucket->offset;
        const int *v_dim = state->affine_cones.v_dim + bucket->offset;
        if (bucket->method == PROJ_METHOD_GRID || bucket->method == PROJ_METHOD_GRID_WEIGHTED)
        {
            int blocks_per_cone = PDHCG_LARGE_CONE_BLOCKS_PER_CONE;
            CUDA_CHECK(cudaMemsetAsync(complementarity, 0, (size_t)bucket->count * sizeof(double)));
            prepare_affine_cone_residuals_grid_kernel<<<bucket->count * blocks_per_cone,
                                                        threads,
                                                        (size_t)threads * sizeof(double)>>>(projection_point,
                                                                                            complementarity,
                                                                                            state->primal_product,
                                                                                            state->affine_cone_offset,
                                                                                            state->pdhg_dual_solution,
                                                                                            start_idx,
                                                                                            v_dim,
                                                                                            bucket->count,
                                                                                            blocks_per_cone);
            finish_affine_cone_complementarity_kernel<<<(bucket->count + threads - 1) / threads, threads>>>(
                complementarity, state->constraint_bound_rescaling, bucket->count);
        }
        else
        {
            prepare_affine_cone_residuals_kernel<<<bucket->count, threads, (size_t)threads * sizeof(double)>>>(
                projection_point,
                complementarity,
                state->primal_product,
                state->affine_cone_offset,
                state->pdhg_dual_solution,
                start_idx,
                v_dim,
                state->constraint_bound_rescaling,
                bucket->count);
        }
    }
}
void pdhcg_device_finish_affine_residuals(
    double *r, const double *p, const double *o, const double *s, double *d, const double *ds, int n)
{
    if (n > 0)
        finish_affine_cone_residuals_kernel<<<(n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(r, p, o, s, d, ds, n);
}
