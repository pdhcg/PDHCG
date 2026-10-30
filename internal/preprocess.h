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
    /* Copy canonical problem arrays to the current device. Descriptors stay on
       the host; cone and quadratic-constraint metadata are borrowed from problem,
       which must outlive the copy. Missing starts and matrices remain NULL.
       Release with free_device_problem, not qp_problem_free. */
    qp_problem_t *convert_problem_to_device(const qp_problem_t *problem, grid_context_t *grid_context);
    void free_device_problem(qp_problem_t *problem);

    /* Borrow device arrays and create unscaled Q/D representations.
       source supplies host metadata; quad_type is already globally classified. */
    processed_qp_problem_t *preprocess_qp_problem(const qp_problem_t *source,
                                                  const qp_problem_t *device_problem,
                                                  quad_obj_type_t quad_type,
                                                  int variable_start);
    void pdhcg_device_preprocess_qp_problem(const qp_problem_t *problem,
                                            int variable_start,
                                            processed_qp_problem_t *processed);
    void free_processed_qp_problem(processed_qp_problem_t *processed);

#ifdef __cplusplus
}
#endif
