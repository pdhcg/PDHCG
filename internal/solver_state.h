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
#include "pdhcg.h"
#include "pdhcg_types.h"
#include "preprocess.h"
#include "solver.h"
#include "utils.h"

#ifdef __cplusplus
extern "C"
{
#endif
    int get_global_n(pdhg_solver_state_t *state);
    int get_n_start(grid_context_t *ctx);
    void initialize_step_size_and_primal_weight(pdhg_solver_state_t *state, const pdhg_parameters_t *params);

    /* Preprocess device_problem, create state storage, then scale it in place. */
    pdhg_solver_state_t *initialize_solver_state(const pdhg_parameters_t *params,
                                                 const qp_problem_t *working_problem,
                                                 const qp_problem_t *device_problem,
                                                 grid_context_t *grid_context);

    void pdhg_solver_state_free(pdhg_solver_state_t *state);

#ifdef __cplusplus
}
#endif
