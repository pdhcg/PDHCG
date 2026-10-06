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

#include "cone_kernel_ops.h"
#include "device_cones.h"
#include <math.h>

cone_proj_method_t pdhcg_device_cone_method(const cone_blocks_t *cones, int bucket, const double *rescaling)
{
    (void)cones;
    (void)bucket;
    (void)rescaling;
    return PROJ_METHOD_THREAD;
}

void launch_cone_reflection(
    cone_proj_method_t method, double *ref, const double *pr, const double *cur, const int *si, const int *vd, int n)
{
    (void)method;
#pragma omp parallel for schedule(static) if (n >= 16)
    for (int k = 0; k < n; ++k)
        for (int i = si[k]; i < si[k] + vd[k] + 2; ++i)
            ref[i] = 2 * pr[i] - cur[i];
}
void launch_cone_dual_slack(
    cone_proj_method_t method, double *slack, const double *obj, const double *dp, const int *si, const int *vd, int n)
{
    (void)method;
#pragma omp parallel for schedule(static) if (n >= 16)
    for (int k = 0; k < n; ++k)
        for (int i = si[k]; i < si[k] + vd[k] + 2; ++i)
            slack[i] = obj[i] - dp[i];
}
void pdhcg_device_prepare_affine_residuals(pdhg_solver_state_t *state, double *point)
{
    for (int b = 0; b < state->affine_cones.num_buckets; ++b)
    {
        const cone_bucket_t *bucket = &state->affine_cones.buckets[b];
#pragma omp parallel for schedule(static) if (bucket->count >= 16)
        for (int k = 0; k < bucket->count; ++k)
        {
            int c = bucket->offset + k, start = state->affine_cones.start_idx[c],
                length = state->affine_cones.v_dim[c] + 2;
            double dot = 0;
            for (int i = start; i < start + length; ++i)
            {
                point[i] = -state->pdhg_dual_solution[i];
                dot += state->pdhg_dual_solution[i] * (state->primal_product[i] + state->affine_cone_offset[i]);
            }
            state->affine_cones.complementarity_residual[c] = fabs(dot) / state->constraint_bound_rescaling;
        }
    }
}
void pdhcg_device_finish_affine_residuals(double *residual,
                                          const double *product,
                                          const double *offset,
                                          const double *rescaling,
                                          double *dual_membership,
                                          const double *dual_rescaling,
                                          int n)
{
#pragma omp parallel for schedule(static) if (n >= 1024)
    for (int i = 0; i < n; ++i)
    {
        residual[i] = (product[i] + offset[i] - residual[i]) * rescaling[i];
        dual_membership[i] *= dual_rescaling[i];
    }
}
