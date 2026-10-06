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
cone_proj_method_t pdhcg_device_cone_method(const cone_blocks_t *cones, int cone, const double *coordinate_rescaling)
{
    cone_type_t type = cones->type[cone];
    int v_dim = cones->v_dim[cone];
    if (type == CONE_EXPONENTIAL || type == CONE_POWER || type == CONE_PSD)
        return PROJ_METHOD_THREAD;
    if (v_dim < 32)
        return PROJ_METHOD_THREAD;
    if (type != CONE_STANDARD_SOC && type != CONE_ROTATED_SOC)
        return PROJ_METHOD_WARP;

    int start = cones->start_idx[cone];
    if (cones->is_fixed)
    {
        for (int slot = 0; slot < v_dim + 2; ++slot)
            if (cones->is_fixed[start + slot])
                return v_dim >= PDHCG_LARGE_CONE_MIN_VDIM ? PROJ_METHOD_GRID_WEIGHTED : PROJ_METHOD_BLOCK;
    }
    if (v_dim < PDHCG_LARGE_CONE_MIN_VDIM || !coordinate_rescaling)
        return PROJ_METHOD_WARP;

    int endpoint0 = start + v_dim;
    int endpoint1 = endpoint0 + 1;

    double d0 = coordinate_rescaling[endpoint0];
    double d1 = coordinate_rescaling[endpoint1];
    if (!(d0 > 0.0) || !(d1 > 0.0) || !isfinite(d0) || !isfinite(d1))
        return PROJ_METHOD_WARP;

    double d_vector = d1;
    if (type == CONE_STANDARD_SOC)
    {
        if (d0 != d1)
            return PROJ_METHOD_GRID_WEIGHTED;
    }
    else
    {
        double d_ref = coordinate_rescaling[start];
        bool scalar_uniform = d0 == d_ref && d1 == d_ref;
        for (int i = 1; i < v_dim && scalar_uniform; ++i)
            scalar_uniform = coordinate_rescaling[start + i] == d_ref;
        if (scalar_uniform)
            return PROJ_METHOD_GRID;
        d_vector = sqrt(d0) * sqrt(d1);
    }

    for (int i = 0; i < v_dim; ++i)
    {
        if (coordinate_rescaling[start + i] != d_vector)
            return PROJ_METHOD_GRID_WEIGHTED;
    }
    return PROJ_METHOD_GRID;
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
