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

#ifndef PDHCG_DISTRIBUTED_INTERFACE_H
#define PDHCG_DISTRIBUTED_INTERFACE_H

#include "pdhcg_types.h"
#include <stdbool.h>

#ifdef __cplusplus
extern "C"
{
#endif

    struct grid_context_s;
    typedef struct grid_context_s grid_context_t;

    typedef enum
    {
        PDHCG_SCOPE_GLOBAL,
        PDHCG_SCOPE_ROW,
        PDHCG_SCOPE_COL
    } pdhcg_comm_scope_t;

    typedef enum
    {
        PDHCG_OP_SUM,
        PDHCG_OP_MAX
    } pdhcg_reduce_op_t;

    void pdhcg_all_reduce_array(
        grid_context_t *ctx, double *buf, int count, pdhcg_reduce_op_t op, pdhcg_comm_scope_t scope, void *stream);

    void pdhcg_all_reduce_scalar(
        grid_context_t *ctx, double *value, pdhcg_reduce_op_t op, pdhcg_comm_scope_t scope, bool on_device);

    int pdhcg_get_grid_p_col(struct grid_context_s *ctx);

    int pdhcg_get_grid_row_coord(struct grid_context_s *ctx);

    int pdhcg_get_global_num_variables(grid_context_t *ctx);

    int pdhcg_get_variable_start(grid_context_t *ctx);

    int pdhcg_get_global_num_cones(grid_context_t *ctx);

    int pdhcg_get_global_num_affine_cones(grid_context_t *ctx);

    pdhcg_result_t *pdhcg_distributed_optimize(const pdhg_parameters_t *params, const qp_problem_t *original_problem);

#ifdef __cplusplus
}
#endif

#endif // PDHCG_DISTRIBUTED_INTERFACE_H
