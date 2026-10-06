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

#pragma once

#include "internal_types.h"

#ifdef __cplusplus
extern "C"
{
#endif

    void initialize_split_cones(pdhg_solver_state_t *state, const rescale_info_t *rescale_info);
    /* Fill the local entries of a host mask using whole-cone fixed metadata. */
    void initialize_split_cone_infeasibility_type(const pdhg_solver_state_t *state, unsigned char *host_type);
    void free_split_cones(pdhg_solver_state_t *state);
    void project_split_cones(pdhg_solver_state_t *state, cone_runtime_t *runtime, double *vector);
    void recompute_split_cone_reflected(pdhg_solver_state_t *state,
                                        double *reflected_primal,
                                        const double *pdhg_primal,
                                        const double *current_primal);
    void compute_split_cone_dual_residual(pdhg_solver_state_t *state, const double *effective_objective);
    double get_split_cone_complementarity_norm(pdhg_solver_state_t *state, norm_type_t norm);
    void prepare_split_affine_cone_residuals(pdhg_solver_state_t *state,
                                             double *projection_point,
                                             const double *primal_product,
                                             const double *affine_cone_offset,
                                             const double *dual_solution);
    void finalize_split_affine_cone_complementarity(pdhg_solver_state_t *state);
    double get_split_affine_cone_complementarity_norm(pdhg_solver_state_t *state, norm_type_t norm);
    void set_split_cone_dual_slack(pdhg_solver_state_t *state,
                                   double *dual_slack,
                                   const double *effective_objective,
                                   const double *dual_product);

#ifdef __cplusplus
}
#endif
