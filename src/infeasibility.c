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

#include "infeasibility.h"
#include "cone_dispatch.h"
#include "device_kernels.h"
#include "pdhg_core_op.h"
#include "utils.h"
#include <math.h>

static bool has_affine_cones(const pdhg_solver_state_t *state)
{
    /* Include the global count so all MPI ranks take the same collective path. */
    return state->affine_cones.num_blocks > 0 || state->affine_cones.split ||
        pdhcg_get_global_num_affine_cones(state->grid_context) > 0;
}

static void project_row_recession(pdhg_solver_state_t *state, double *vector)
{
    pdhcg_device_primal_infeasibility_project(
        vector, state->constraint_lower_bound, state->constraint_upper_bound, NULL, state->num_constraints);
    project_cone_runtime(state, &state->affine_cones, vector, state->affine_cones.infeasibility_workspace);
}

static void project_dual_ray(pdhg_solver_state_t *state, bool conic_rows)
{
    if (!conic_rows)
    {
        pdhcg_device_dual_infeasibility_project(state->delta_dual_solution,
                                                state->constraint_lower_bound,
                                                state->constraint_upper_bound,
                                                state->num_constraints);
        return;
    }

    /* For Ax+b in D, the lower support is finite on rec(D)*. Moreau gives
       Pi_rec(D)*(y) = y + Pi_rec(D)(-y), including non-self-dual cones. */
    DEVICE_CHECK(pdhcg_device_copy(state->primal_slack,
                                   state->delta_dual_solution,
                                   state->num_constraints * sizeof(double),
                                   PDHCG_COPY_DEVICE_TO_DEVICE));
    const double negative_one = -1.0;
    DEVICE_CHECK(pdhcg_device_scal(state->blas_handle, state->num_constraints, &negative_one, state->primal_slack, 1));
    project_row_recession(state, state->primal_slack);
    DEVICE_CHECK(pdhcg_device_axpy(
        state->blas_handle, state->num_constraints, &HOST_ONE, state->primal_slack, 1, state->delta_dual_solution, 1));
}

static void compute_variable_support(pdhg_solver_state_t *state)
{
    if (!state->has_variable_cones)
    {
        pdhcg_device_dual_objective_dual_slack_contribution_array(state->dual_product,
                                                                  state->infeasibility_dual_workspace,
                                                                  state->variable_lower_bound_finite_val,
                                                                  state->variable_upper_bound_finite_val,
                                                                  state->num_variables);
        return;
    }

    /* For a fixed section C, q = Pi_C(p + A'y) and s = -A'y + q - p
       satisfy inf_{z in C} s'z = s'q. Thus p-q is the stationarity error.
       Homogeneous blocks use p=0 (Moreau); ordinary boxes stay analytic.
       All operations use scaled coordinates, including pinned fixed bounds. */
    pdhcg_device_prepare_infeasibility_projection(state->infeasibility_dual_workspace,
                                                  state->dual_product,
                                                  state->pdhg_primal_solution,
                                                  state->variable_lower_bound,
                                                  state->variable_upper_bound,
                                                  state->cones.infeasibility_type,
                                                  state->num_variables);
    project_cone_runtime(
        state, &state->cones, state->infeasibility_dual_workspace, state->cones.infeasibility_workspace);
    pdhcg_device_finish_infeasibility_projection(state->dual_product,
                                                 state->infeasibility_dual_workspace,
                                                 state->pdhg_primal_solution,
                                                 state->variable_lower_bound,
                                                 state->variable_upper_bound,
                                                 state->cones.infeasibility_type,
                                                 state->variable_rescaling,
                                                 state->num_variables);
}

static double global_inf_norm(pdhg_solver_state_t *state, const double *vector, int length, pdhcg_comm_scope_t scope)
{
    double norm = get_vector_inf_norm(state->blas_handle, length, vector);
    pdhcg_all_reduce_scalar(state->grid_context, &norm, PDHCG_OP_MAX, scope, false);
    return norm;
}

static void normalize_candidate(pdhg_solver_state_t *state, double *vector, int length, pdhcg_comm_scope_t scope)
{
    double norm = global_inf_norm(state, vector, length, scope);
    if (isfinite(norm) && norm > 0.0)
    {
        double scale = 1.0 / norm;
        DEVICE_CHECK(pdhcg_device_scal(state->blas_handle, length, &scale, vector, 1));
    }
}

static bool certificate_meets_tolerance(double residual, double objective, double tolerance)
{
    return isfinite(residual) && residual >= 0.0 && isfinite(objective) && objective > 0.0 &&
        residual / objective <= tolerance;
}

termination_reason_t check_infeasibility_criteria(const pdhg_solver_state_t *state, double tolerance)
{
    if (certificate_meets_tolerance(state->max_dual_ray_infeasibility, state->dual_ray_objective, tolerance))
        return TERMINATION_REASON_PRIMAL_INFEASIBLE;
    if (certificate_meets_tolerance(
            state->max_primal_ray_infeasibility, -state->primal_ray_linear_objective, tolerance))
        return TERMINATION_REASON_DUAL_INFEASIBLE;
    return TERMINATION_REASON_UNSPECIFIED;
}

static void evaluate_candidates(pdhg_solver_state_t *state)
{
    bool conic_rows = has_affine_cones(state);

    /* Cone root solves have finite absolute accuracy. Normalize homogeneous
       candidates before projection, not just their final reported statistics. */
    if (state->has_variable_cones || conic_rows)
    {
        normalize_candidate(state, state->delta_primal_solution, state->num_variables, PDHCG_SCOPE_ROW);
        normalize_candidate(state, state->delta_dual_solution, state->num_constraints, PDHCG_SCOPE_COL);
    }

    /* Delta vectors and matrix products are scratch until the next update or
       restart. Keep dual_slack intact: LP residual evaluation reads it. */
    pdhcg_device_primal_infeasibility_project(state->delta_primal_solution,
                                              state->variable_lower_bound,
                                              state->variable_upper_bound,
                                              state->cones.infeasibility_type,
                                              state->num_variables);
    /* Pinned variable bounds first set fixed direction coordinates to zero;
       the same fixed-section projector then projects onto rec(C). */
    if (state->has_variable_cones)
        project_cone_runtime(state, &state->cones, state->delta_primal_solution, state->cones.infeasibility_workspace);
    project_dual_ray(state, conic_rows);

    double primal_ray_inf_norm =
        global_inf_norm(state, state->delta_primal_solution, state->num_variables, PDHCG_SCOPE_ROW);

    if (primal_ray_inf_norm > 0.0)
    {
        double scale = 1.0 / primal_ray_inf_norm;
        DEVICE_CHECK(
            pdhcg_device_scal(state->blas_handle, state->num_variables, &scale, state->delta_primal_solution, 1));
    }

    double dual_ray_inf_norm =
        global_inf_norm(state, state->delta_dual_solution, state->num_constraints, PDHCG_SCOPE_COL);

    pdhcg_spmv_execute(state->sparse_handle,
                       state->spmv_ctx_A,
                       &HOST_ONE,
                       &HOST_ZERO,
                       state->delta_primal_solution,
                       state->primal_product);

    pdhcg_all_reduce_array(
        state->grid_context, state->primal_product, state->num_constraints, PDHCG_OP_SUM, PDHCG_SCOPE_ROW, 0);

    pdhcg_spmv_execute(state->sparse_handle,
                       state->spmv_ctx_At,
                       &HOST_ONE,
                       &HOST_ZERO,
                       state->delta_dual_solution,
                       state->dual_product);

    pdhcg_all_reduce_array(
        state->grid_context, state->dual_product, state->num_variables, PDHCG_OP_SUM, PDHCG_SCOPE_COL, 0);

    DEVICE_CHECK(pdhcg_device_dot(state->blas_handle,
                                  state->num_variables,
                                  state->objective_vector,
                                  1,
                                  state->delta_primal_solution,
                                  1,
                                  &state->primal_ray_linear_objective));

    pdhcg_all_reduce_scalar(
        state->grid_context, &state->primal_ray_linear_objective, PDHCG_OP_SUM, PDHCG_SCOPE_ROW, false);
    state->primal_ray_linear_objective /= (state->constraint_bound_rescaling * state->objective_vector_rescaling);

    pdhcg_device_dual_solution_dual_objective_contribution(state->constraint_lower_bound_finite_val,
                                                           state->constraint_upper_bound_finite_val,
                                                           state->affine_cone_offset,
                                                           state->delta_dual_solution,
                                                           state->num_constraints,
                                                           state->primal_slack);

    compute_variable_support(state);

    double sum_primal_slack =
        get_vector_sum(state->blas_handle, state->num_constraints, state->ones_dual, state->primal_slack);

    pdhcg_all_reduce_scalar(state->grid_context, &sum_primal_slack, PDHCG_OP_SUM, PDHCG_SCOPE_COL, false);

    double sum_dual_slack =
        get_vector_sum(state->blas_handle, state->num_variables, state->ones_primal, state->infeasibility_dual_workspace);

    pdhcg_all_reduce_scalar(state->grid_context, &sum_dual_slack, PDHCG_OP_SUM, PDHCG_SCOPE_ROW, false);

    state->dual_ray_objective =
        (sum_primal_slack + sum_dual_slack) / (state->constraint_bound_rescaling * state->objective_vector_rescaling);

    if (conic_rows)
    {
        /* Offsets disappear in rec(D). Compare Ad to its recession projection. */
        DEVICE_CHECK(pdhcg_device_copy(state->primal_slack,
                                       state->primal_product,
                                       state->num_constraints * sizeof(double),
                                       PDHCG_COPY_DEVICE_TO_DEVICE));
        project_row_recession(state, state->primal_slack);
        pdhcg_device_projection_residual(
            state->primal_slack, state->primal_product, state->constraint_rescaling, state->num_constraints);
    }
    else
        pdhcg_device_compute_primal_infeasibility(state->primal_product,
                                                  state->constraint_lower_bound,
                                                  state->constraint_upper_bound,
                                                  state->num_constraints,
                                                  state->primal_slack,
                                                  state->constraint_rescaling);
    if (!state->has_variable_cones)
        pdhcg_device_compute_dual_infeasibility(state->dual_product,
                                                state->variable_lower_bound,
                                                state->variable_upper_bound,
                                                state->num_variables,
                                                state->infeasibility_dual_workspace,
                                                state->variable_rescaling);

    /* The kernels restore diagonal row/column scaling. Restore the global
       bound/objective factors too, so residuals and objectives use the same
       original-problem units. */
    state->max_primal_ray_infeasibility =
        global_inf_norm(state, state->primal_slack, state->num_constraints, PDHCG_SCOPE_COL) /
        state->constraint_bound_rescaling;
    double dual_slack_norm =
        global_inf_norm(state,
                        state->has_variable_cones ? state->dual_product : state->infeasibility_dual_workspace,
                        state->num_variables,
                        PDHCG_SCOPE_ROW);
    state->max_dual_ray_infeasibility = dual_slack_norm / state->objective_vector_rescaling;

    if (state->problem_type != LP && state->quadratic_objective_term->quad_obj_type != PDHCG_NON_Q)
    {
        update_obj_product(state, state->delta_primal_solution);
        /* Q_hat = (objective_scale / bound_scale) V^-1 Q V^-1 and
           d = V^-1 d_hat / bound_scale, hence Qd = V Q_hat d_hat / objective_scale. */
        pdhcg_device_element_wise_mul(state->quadratic_objective_term->primal_obj_product,
                                      state->variable_rescaling,
                                      state->infeasibility_dual_workspace,
                                      state->num_variables);
        double q_ray_norm =
            global_inf_norm(state, state->infeasibility_dual_workspace, state->num_variables, PDHCG_SCOPE_ROW) /
            state->objective_vector_rescaling;
        state->max_primal_ray_infeasibility = fmax(state->max_primal_ray_infeasibility, q_ray_norm);
    }

    /* A common ray normalization cancels in the certificate ratio. Keep its
       raw norm separate from the residual expressed in original units. */
    double scaling_factor = fmax(dual_ray_inf_norm, dual_slack_norm);
    if (scaling_factor > 0.0)
    {
        state->max_dual_ray_infeasibility /= scaling_factor;
        state->dual_ray_objective /= scaling_factor;
    }
    else
    {
        state->max_dual_ray_infeasibility = 0.0;
        state->dual_ray_objective = 0.0;
    }
}

static double certificate_ratio(double residual, double objective)
{
    return isfinite(residual) && residual >= 0.0 && isfinite(objective) && objective > 0.0 ? residual / objective
                                                                                           : INFINITY;
}

void compute_infeasibility_information(pdhg_solver_state_t *state)
{
    evaluate_candidates(state);
    if (!state->has_variable_cones && !has_affine_cones(state))
        return;

    double primal_residual = state->max_primal_ray_infeasibility;
    double primal_objective = state->primal_ray_linear_objective;
    double dual_residual = state->max_dual_ray_infeasibility;
    double dual_objective = state->dual_ray_objective;

    /* On divergent conic problems, subtracting large neighboring iterates can
       lose the certificate direction. The iterates themselves are independent
       candidates: accept them only through the same complete certificate tests. */
    DEVICE_CHECK(pdhcg_device_copy(state->delta_primal_solution,
                                   state->pdhg_primal_solution,
                                   state->num_variables * sizeof(double),
                                   PDHCG_COPY_DEVICE_TO_DEVICE));
    DEVICE_CHECK(pdhcg_device_copy(state->delta_dual_solution,
                                   state->pdhg_dual_solution,
                                   state->num_constraints * sizeof(double),
                                   PDHCG_COPY_DEVICE_TO_DEVICE));
    evaluate_candidates(state);

    if (certificate_ratio(primal_residual, -primal_objective) <=
        certificate_ratio(state->max_primal_ray_infeasibility, -state->primal_ray_linear_objective))
    {
        state->max_primal_ray_infeasibility = primal_residual;
        state->primal_ray_linear_objective = primal_objective;
    }
    if (certificate_ratio(dual_residual, dual_objective) <=
        certificate_ratio(state->max_dual_ray_infeasibility, state->dual_ray_objective))
    {
        state->max_dual_ray_infeasibility = dual_residual;
        state->dual_ray_objective = dual_objective;
    }
}
