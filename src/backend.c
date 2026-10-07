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

#include "backend.h"
#include "cbf_parser.h"
#include "cone_utils.h"
#include "device_general_op.h"
#include "mps_parser.h"
#include "utils.h"

const pdhcg_backend_t *pdhcg_backend(void)
{
    static const pdhcg_backend_t backend = {
        .create = create_qp_problem,
        .set_start = set_start_values,
        .set_fixed = set_cone_fixed,
        .to_socp = qcqp_to_socp_qp,
        .solve = solve_qp_problem,
        .solve_distributed = solve_qp_problem_distributed,
        .defaults = set_default_parameters,
        .validate = pdhcg_validate_parameters,
        .free_result = pdhcg_result_free,
        .free_problem = qp_problem_free,
        .read_mps = read_mps_file,
        .read_cbf = read_cbf_file,
        .presolve_version = pdhcg_presolve_version,
        .presolve_available = pdhcg_presolve_available,
        .presolve_status = pdhcg_get_presolve_status_str,
        .presolve = pdhcg_presolve,
        .presolve_result = pdhcg_create_result_from_presolve,
        .postsolve = pdhcg_postsolve,
        .free_presolve = pdhcg_presolve_info_free,
        .allocate_host = safe_malloc,
        .termination_name = termination_reason_to_string,
        .cone_size = cone_length,
        .cone_block_size = cone_block_length,
        .initialize = pdhcg_device_initialize,
    };
    return &backend;
}
