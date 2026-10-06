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

#pragma once
#include "internal_types.h"
#ifdef __cplusplus
extern "C"
{
#endif
    cone_proj_method_t pdhcg_device_cone_method(const cone_blocks_t *cones, int cone, const double *rescaling);
    void pdhcg_device_prepare_affine_residuals(pdhg_solver_state_t *state, double *point);
    void pdhcg_device_finish_affine_residuals(double *residual,
                                              const double *product,
                                              const double *offset,
                                              const double *rescaling,
                                              double *dual_membership,
                                              const double *dual_rescaling,
                                              int n);
    void launch_power_cone_primal_violation(double *absolute,
                                            double *relative,
                                            const double *primal,
                                            const double *rescaling,
                                            const int *start,
                                            const double *alpha,
                                            double bound_scale,
                                            int n);
#ifdef __cplusplus
}
#endif
