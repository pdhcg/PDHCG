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

#include "solver.h"
#include "internal_types.h"
#include "pdhcg.h"
#include "pdhg_core_op.h"
#include "infeasibility.h"
#include "preconditioner.h"
#include "presolve_wrapper.h"
#include "qcqp_transform.h"
#include "solver_state.h"
#include "utils.h"
#include <time.h>
#ifdef _WIN32
#include <windows.h>
#endif
#include <math.h>
#include <stdbool.h>
#include <stdio.h>

static double monotonic_time_seconds(void)
{
#ifdef _WIN32
    LARGE_INTEGER counter, frequency;
    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&counter);
    return (double)counter.QuadPart / (double)frequency.QuadPart;
#else
    struct timespec now;
    if (clock_gettime(CLOCK_MONOTONIC, &now) != 0)
    {
        perror("clock_gettime");
        exit(EXIT_FAILURE);
    }
    return (double)now.tv_sec + (double)now.tv_nsec * 1e-9;
#endif
}

pdhcg_result_t *optimize(const pdhg_parameters_t *input_params, const qp_problem_t *original_problem)
{
    pdhg_parameters_t copyed_params = *input_params;
    pdhg_parameters_t *params = &copyed_params;
    const qp_problem_t *input_problem = original_problem;

    print_initial_info(input_params, original_problem);

    qp_problem_t *transformed = NULL;
    if (original_problem->num_quadratic_constraints > 0)
    {
        transformed = qcqp_to_socp_qp(original_problem, params->default_cone_type);
        if (!transformed)
        {
            fprintf(stderr, "Error: QCQP -> SOCP transformation failed; cannot solve.\n");
            return NULL;
        }
        if (params->verbose >= 1)
        {
            const char *form_name = (params->default_cone_type == CONE_STANDARD_SOC) ? "standard" : "rotated";
            fprintf(stderr,
                    "[QCQP] %d quadratic constraint(s) reformulated as %d "
                    "%s SOC block(s); extended problem: %d vars, "
                    "%d rows, %d nnz.\n",
                    original_problem->num_quadratic_constraints,
                    transformed->cones.num_cones,
                    form_name,
                    transformed->num_variables,
                    transformed->num_constraints,
                    transformed->constraint_matrix_num_nonzeros);
        }
        original_problem = transformed;
    }

    pdhcg_presolve_info_t *presolve_info = NULL;
    const qp_problem_t *working_problem = original_problem;
    bool working_problem_needs_free = false;

    if (params->presolve && original_problem->affine_cones.num_cones == 0 && pdhcg_presolve_available())
    {
        presolve_info = pdhcg_presolve(original_problem, params);
        if (presolve_info)
        {
            if (presolve_info->problem_solved_during_presolve)
            {
                pdhcg_result_t *result = pdhcg_create_result_from_presolve(presolve_info, original_problem);
                restore_qcqp_result_dimensions(result, transformed ? input_problem : NULL);
                if (result)
                {
                    pdhg_final_log(result, params);
                }
                pdhcg_presolve_info_free(presolve_info);
                if (transformed)
                {
                    qp_problem_free(transformed);
                }
                return result;
            }

            if (presolve_info->reduced_problem)
            {
                working_problem = presolve_info->reduced_problem;
            }
        }
    }

    if (working_problem->num_constraints == 0 || working_problem->constraint_matrix == NULL)
    {
        working_problem = create_problem_with_dummy_constraint(working_problem);
        working_problem_needs_free = true;
    }

    rescale_info_t *rescale_info = rescale_problem(params, working_problem);
    grid_context_t *grid_context = NULL;
    pdhg_solver_state_t *state = initialize_solver_state(params, working_problem, rescale_info, grid_context);

    if (state->quadratic_objective_term->nonconvexity < 0)
    {
        state->inner_solver->iteration_limit = 1;
    }

    rescale_info_free(rescale_info);
    initialize_step_size_and_primal_weight(state, params);
    const double start_time = monotonic_time_seconds();
    bool do_restart = false;

    while (state->total_count < params->termination_criteria.iteration_limit)
    {
        if ((state->is_this_major_iteration || state->total_count == 0) ||
            (state->total_count % get_print_frequency(state->total_count) == 0))
        {
            /* Conic residuals reuse the delta vectors, so consume the fixed-point
               directions before evaluating ordinary optimality residuals. */
            if (state->is_this_major_iteration)
            {
                compute_infeasibility_information(state);
            }
            compute_residual(state, params->optimality_norm);

            state->cumulative_time_sec = monotonic_time_seconds() - start_time;

            check_termination_criteria(state, &params->termination_criteria);
            display_iteration_stats(state, params->verbose);
            if (state->termination_reason != TERMINATION_REASON_UNSPECIFIED)
            {
                break;
            }
        }

        if ((state->is_this_major_iteration || state->total_count == 0))
        {
            do_restart =
                should_do_adaptive_restart(state, &params->restart_params, params->termination_evaluation_frequency);
            if (do_restart)
                perform_restart(state, params);
        }

        state->is_this_major_iteration = ((state->total_count + 1) % params->termination_evaluation_frequency) == 0;

        pdhg_update(state);

        if (state->is_this_major_iteration || do_restart)
        {
            compute_fixed_point_error(state);
            if (do_restart)
            {
                state->initial_fixed_point_error = state->fixed_point_error;
                do_restart = false;
            }
        }
        halpern_update(state, params->reflection_coefficient);

        state->inner_count++;
        state->total_count++;
    }

    if (state->termination_reason == TERMINATION_REASON_UNSPECIFIED)
    {
        state->termination_reason = TERMINATION_REASON_ITERATION_LIMIT;
        compute_residual(state, params->optimality_norm);
        display_iteration_stats(state, params->verbose);
    }

    // if (params->feasibility_polishing &&
    //     state->termination_reason != TERMINATION_REASON_DUAL_INFEASIBLE &&
    //     state->termination_reason != TERMINATION_REASON_PRIMAL_INFEASIBLE) {
    //   feasibility_polish(params, state);
    // }

    pdhcg_result_t *result = create_result_from_state(state, original_problem);

    if (presolve_info && presolve_info->reduced_problem)
    {
        if (!pdhcg_postsolve(presolve_info, result, original_problem))
        {
            fprintf(stderr, "Error: PreFOS primal-dual postsolve failed.\n");
            result->termination_reason = TERMINATION_REASON_UNSPECIFIED;
        }
    }
    if (working_problem_needs_free)
    {
        qp_problem_free((qp_problem_t *)working_problem);
    }

    restore_qcqp_result_dimensions(result, transformed ? input_problem : NULL);

    pdhg_final_log(result, params);
    pdhg_solver_state_free(state);
    pdhcg_presolve_info_free(presolve_info);
    if (transformed)
    {
        qp_problem_free(transformed);
    }
    DEVICE_CHECK(pdhcg_device_last_error());
    return result;
}
