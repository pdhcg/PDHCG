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

#include "cone_dispatch.h"
#include "cone_kernel_ops.h"
#include "device_cones.h"
#include "device_kernels.h"
#include "distributed_conic.h"
#include "distributed_interface.h"
#include "internal_types.h"
#include "pdhcg.h"

#include "pdhcg_psd_cone.h"
#include "pdhg_core_op.h"
#include "preconditioner.h"
#include "solver.h"
#include "solver_state.h"
#include "spmv_backend.h"
#include "utils.h"
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <time.h>

static const double SPECTRAL_ESTIMATE_STEP_SAFETY = 0.998;

static double safeguarded_spectral_estimate(double estimate)
{
    return fabs(estimate) / SPECTRAL_ESTIMATE_STEP_SAFETY;
}

static const double *cone_dual_residual_effective_obj(pdhg_solver_state_t *state)
{
    if (state->cones.effective_objective_gradient)
        return state->cones.effective_objective_gradient;
    return state->objective_vector;
}

static double cone_residual_norm(pdhg_solver_state_t *state, int count, const double *values, norm_type_t norm)
{
    if (count <= 0)
        return 0.0;
    if (norm == NORM_TYPE_L_INF)
        return get_vector_inf_norm(state->blas_handle, count, values);

    double result = 0.0;
    DEVICE_CHECK(pdhcg_device_nrm2(state->blas_handle, count, values, 1, &result));
    return result;
}

/*
 * Cone membership plus dual-cone membership does not enforce complementarity.
 * This gradient mapping is zero exactly when the current point is feasible and
 * the reduced gradient belongs to the negative normal cone, including for
 * fixed cone cross-sections.
 */
static void augment_conic_projected_gradient_residual(pdhg_solver_state_t *state, const double *effective_obj)
{
    double step_size = state->step_size / state->primal_weight;
    if (!(step_size > 0.0) || !isfinite(step_size))
        step_size = 1.0;

    pdhcg_device_prepare_projected_gradient_point(state->delta_primal_solution,
                                                  state->pdhg_primal_solution,
                                                  effective_obj,
                                                  state->dual_product,
                                                  state->variable_lower_bound,
                                                  state->variable_upper_bound,
                                                  step_size,
                                                  state->num_variables);
    project_cone_runtime(state, &state->cones, state->delta_primal_solution, state->cones.residual_workspace);
    pdhcg_device_augment_projected_gradient_residual(state->dual_residual,
                                                     state->pdhg_primal_solution,
                                                     state->delta_primal_solution,
                                                     state->variable_rescaling,
                                                     step_size,
                                                     state->num_variables);
}

static double compute_cone_complementarity_norm(pdhg_solver_state_t *state, norm_type_t norm)
{
    double residual_norm =
        cone_residual_norm(state, state->cones.num_blocks, state->cones.complementarity_residual, norm);

    double distributed_norm = get_split_cone_complementarity_norm(state, norm);
    residual_norm =
        norm == NORM_TYPE_L_INF ? fmax(residual_norm, distributed_norm) : hypot(residual_norm, distributed_norm);
    return residual_norm;
}

static bool has_affine_cone_constraints(const pdhg_solver_state_t *state)
{
    return state->affine_cones.num_blocks > 0 || state->affine_cones.split != NULL ||
        pdhcg_get_global_num_affine_cones(state->grid_context) > 0;
}

static void compute_affine_cone_residuals(pdhg_solver_state_t *state,
                                          norm_type_t norm,
                                          double *dual_membership_norm,
                                          double *complementarity_norm)
{
    *dual_membership_norm = 0.0;
    *complementarity_norm = 0.0;

    if (!has_affine_cone_constraints(state))
        return;

    int rows = state->num_constraints;
    double *projection_point = state->delta_dual_solution;
    if (state->affine_cones.num_blocks > 0)
    {
        pdhcg_device_prepare_affine_residuals(state, projection_point);
        prepare_psd_affine_cone_residuals(state->affine_cones.psd,
                                          projection_point,
                                          state->affine_cones.complementarity_residual,
                                          state->primal_product,
                                          state->affine_cone_offset,
                                          state->pdhg_dual_solution,
                                          state->constraint_bound_rescaling);
    }
    prepare_split_affine_cone_residuals(
        state, projection_point, state->primal_product, state->affine_cone_offset, state->pdhg_dual_solution);

    project_cone_runtime(state, &state->affine_cones, projection_point, state->affine_cones.residual_workspace);
    if (rows > 0)
    {
        pdhcg_device_finish_affine_residuals(state->primal_residual,
                                             state->primal_product,
                                             state->affine_cone_offset,
                                             state->constraint_rescaling,
                                             projection_point,
                                             state->affine_cones.coordinate_rescaling,
                                             rows);
    }
    finalize_split_affine_cone_complementarity(state);

    if (norm == NORM_TYPE_L_INF)
    {
        *dual_membership_norm = cone_residual_norm(state, rows, projection_point, norm);
        *complementarity_norm = cone_residual_norm(
            state, state->affine_cones.num_blocks, state->affine_cones.complementarity_residual, norm);
        *complementarity_norm = fmax(*complementarity_norm, get_split_affine_cone_complementarity_norm(state, norm));
        pdhcg_all_reduce_scalar(state->grid_context, dual_membership_norm, PDHCG_OP_MAX, PDHCG_SCOPE_COL, false);
        pdhcg_all_reduce_scalar(state->grid_context, complementarity_norm, PDHCG_OP_MAX, PDHCG_SCOPE_COL, false);
    }
    else
    {
        *dual_membership_norm = cone_residual_norm(state, rows, projection_point, norm);
        *complementarity_norm = cone_residual_norm(
            state, state->affine_cones.num_blocks, state->affine_cones.complementarity_residual, norm);
        double membership_squared = *dual_membership_norm * *dual_membership_norm;
        double complementarity_squared = *complementarity_norm * *complementarity_norm;
        double split_norm = get_split_affine_cone_complementarity_norm(state, norm);
        complementarity_squared += split_norm * split_norm;
        pdhcg_all_reduce_scalar(state->grid_context, &membership_squared, PDHCG_OP_SUM, PDHCG_SCOPE_COL, false);
        pdhcg_all_reduce_scalar(state->grid_context, &complementarity_squared, PDHCG_OP_SUM, PDHCG_SCOPE_COL, false);
        *dual_membership_norm = sqrt(membership_squared);
        *complementarity_norm = sqrt(complementarity_squared);
    }
}

static void compute_power_cone_primal_violation(pdhg_solver_state_t *state,
                                                norm_type_t optimality_norm,
                                                double *absolute_violation,
                                                double *relative_violation)
{
    *absolute_violation = 0.0;
    *relative_violation = 0.0;
    if (!state->cones.has_power_cones)
        return;

    double absolute_accumulator = 0.0;
    double relative_accumulator = 0.0;
    for (int b = 0; b < state->cones.num_buckets; ++b)
    {
        const cone_bucket_t *bucket = &state->cones.buckets[b];
        if (bucket->type != CONE_POWER)
            continue;
        double *absolute_workspace = state->cones.power_violation_workspace + bucket->offset;
        double *relative_workspace = state->cones.power_violation_workspace + state->cones.num_blocks + bucket->offset;
        launch_power_cone_primal_violation(absolute_workspace,
                                           relative_workspace,
                                           state->pdhg_primal_solution,
                                           state->variable_rescaling,
                                           state->cones.start_idx + bucket->offset,
                                           state->cones.power_alpha + bucket->offset,
                                           state->constraint_bound_rescaling,
                                           bucket->count);
        if (optimality_norm == NORM_TYPE_L_INF)
        {
            absolute_accumulator = fmax(absolute_accumulator,
                                        cone_residual_norm(state, bucket->count, absolute_workspace, optimality_norm));
            relative_accumulator = fmax(relative_accumulator,
                                        cone_residual_norm(state, bucket->count, relative_workspace, optimality_norm));
        }
        else
        {
            double bucket_absolute_norm = cone_residual_norm(state, bucket->count, absolute_workspace, optimality_norm);
            double bucket_relative_norm = cone_residual_norm(state, bucket->count, relative_workspace, optimality_norm);
            absolute_accumulator += bucket_absolute_norm * bucket_absolute_norm;
            relative_accumulator += bucket_relative_norm * bucket_relative_norm;
        }
    }
    if (optimality_norm == NORM_TYPE_L_INF)
    {
        pdhcg_all_reduce_scalar(state->grid_context, &absolute_accumulator, PDHCG_OP_MAX, PDHCG_SCOPE_ROW, false);
        pdhcg_all_reduce_scalar(state->grid_context, &relative_accumulator, PDHCG_OP_MAX, PDHCG_SCOPE_ROW, false);
    }
    else
    {
        pdhcg_all_reduce_scalar(state->grid_context, &absolute_accumulator, PDHCG_OP_SUM, PDHCG_SCOPE_ROW, false);
        pdhcg_all_reduce_scalar(state->grid_context, &relative_accumulator, PDHCG_OP_SUM, PDHCG_SCOPE_ROW, false);
        absolute_accumulator = sqrt(absolute_accumulator);
        relative_accumulator = sqrt(relative_accumulator);
    }
    *absolute_violation = absolute_accumulator / state->constraint_bound_rescaling;
    *relative_violation = relative_accumulator;
}

static void apply_lowrank_middle(pdhcg_device_blas_t blas_handle, quadratic_objective_term_t *qot)
{
    int rank = qot->num_rank_lowrank_obj;
    if (qot->lowrank_middle_type == 0 || rank <= 0)
        return;

    if (qot->lowrank_middle_type == 1)
    {
        pdhcg_device_element_wise_mul_inplace(qot->Rx_product, qot->d_middle_diag, rank);
        return;
    }

    pdhcg_device_pointer_mode_t prev_mode;
    DEVICE_CHECK(pdhcg_device_get_pointer_mode(blas_handle, &prev_mode));
    DEVICE_CHECK(pdhcg_device_set_pointer_mode(blas_handle, PDHCG_POINTER_HOST));
    DEVICE_CHECK(pdhcg_device_symv(blas_handle,
                                   PDHCG_TRIANGLE_LOWER,
                                   rank,
                                   &HOST_ONE,
                                   qot->d_middle_dense,
                                   rank,
                                   qot->Rx_product,
                                   1,
                                   &HOST_ZERO,
                                   qot->Rx_buffer,
                                   1));
    DEVICE_CHECK(pdhcg_device_set_pointer_mode(blas_handle, prev_mode));
    DEVICE_CHECK(pdhcg_device_copy_async(
        qot->Rx_product, qot->Rx_buffer, (size_t)rank * sizeof(double), PDHCG_COPY_DEVICE_TO_DEVICE));
}

static void compute_quadratic_objective_product(pdhcg_device_sparse_t sparse_handle,
                                                pdhcg_device_blas_t blas_handle,
                                                quadratic_objective_term_t *qot,
                                                double *primal_solution,
                                                int num_variables,
                                                grid_context_t *grid_context)
{
    if (qot->quad_obj_type == PDHCG_NON_Q)
        return;

    if (qot->quad_obj_type == PDHCG_DIAG_Q)
    {
        pdhcg_device_element_wise_mul(
            qot->diagonal_objective_matrix, primal_solution, qot->primal_obj_product, num_variables);
        return;
    }

    bool has_sparse = quadratic_type_has_sparse_component(qot->quad_obj_type);
    bool has_lowrank = quadratic_type_has_lowrank_component(qot->quad_obj_type);
    if (!has_sparse && !has_lowrank)
    {
        fprintf(stderr, "Error: Unknown Quadratic Objective Type detected.\n");
        exit(EXIT_FAILURE);
    }

    if (has_sparse)
    {
        int global_num_variables = pdhcg_get_global_num_variables(grid_context);
        if (global_num_variables <= 0)
            global_num_variables = num_variables;
        pdhcg_spmv_execute(
            sparse_handle, qot->spmv_ctx_Q, &HOST_ONE, &HOST_ZERO, primal_solution, qot->global_primal_obj_product);
        pdhcg_all_reduce_array(
            grid_context, qot->global_primal_obj_product, global_num_variables, PDHCG_OP_SUM, PDHCG_SCOPE_ROW, 0);
    }

    if (has_lowrank)
    {
        pdhcg_spmv_execute(sparse_handle, qot->spmv_ctx_R, &HOST_ONE, &HOST_ZERO, primal_solution, qot->Rx_product);
        pdhcg_all_reduce_array(
            grid_context, qot->Rx_product, qot->num_rank_lowrank_obj, PDHCG_OP_SUM, PDHCG_SCOPE_ROW, 0);
        apply_lowrank_middle(blas_handle, qot);
        pdhcg_spmv_execute(sparse_handle,
                           qot->spmv_ctx_Rt,
                           &HOST_ONE,
                           has_sparse ? &HOST_ONE : &HOST_ZERO,
                           qot->Rx_product,
                           qot->primal_obj_product);
    }
}

void update_obj_product(pdhg_solver_state_t *state, double *primal_solution)
{
    compute_quadratic_objective_product(state->sparse_handle,
                                        state->blas_handle,
                                        state->quadratic_objective_term,
                                        primal_solution,
                                        state->num_variables,
                                        state->grid_context);
}

static void update_cone_effective_objective_gradient(pdhg_solver_state_t *state)
{
    if (!state->cones.effective_objective_gradient)
        return;
    pdhcg_device_vector_add(state->objective_vector,
                            state->quadratic_objective_term->primal_obj_product,
                            state->cones.effective_objective_gradient,
                            state->num_variables);
}

double compute_xQx(pdhg_solver_state_t *state, double *primal_sol, double *primal_obj_product)
{
    if (state->quadratic_objective_term->quad_obj_type == PDHCG_NON_Q)
        return 0.0;

    double xQx = 0.0;
    DEVICE_CHECK(
        pdhcg_device_dot(state->blas_handle, state->num_variables, primal_sol, 1, primal_obj_product, 1, &xQx));
    pdhcg_all_reduce_scalar(state->grid_context, &xQx, PDHCG_OP_SUM, PDHCG_SCOPE_ROW, false);
    return xQx;
}

void lp_primal_update(pdhg_solver_state_t *state, double step_size)
{
    bool force_major_for_cone = state->has_variable_cones;
    if (state->is_this_major_iteration || force_major_for_cone ||
        ((state->total_count + 2) % get_print_frequency(state->total_count + 2)) == 0)
    {
        pdhcg_device_compute_lp_next_pdhg_primal_solution_major(state->current_primal_solution,
                                                                state->pdhg_primal_solution,
                                                                state->reflected_primal_solution,
                                                                state->dual_product,
                                                                state->objective_vector,
                                                                state->variable_lower_bound,
                                                                state->variable_upper_bound,
                                                                state->num_variables,
                                                                step_size,
                                                                state->dual_slack);
    }
    else
    {
        pdhcg_device_compute_lp_next_pdhg_primal_solution(state->current_primal_solution,
                                                          state->reflected_primal_solution,
                                                          state->dual_product,
                                                          state->objective_vector,
                                                          state->variable_lower_bound,
                                                          state->variable_upper_bound,
                                                          state->num_variables,
                                                          step_size);
    }
}

void diag_q_primal_update(pdhg_solver_state_t *state, double step_size)
{
    bool force_major_for_cone = state->has_variable_cones;
    if (state->is_this_major_iteration ||
        ((state->total_count + 2) % get_print_frequency(state->total_count + 2)) == 0 || force_major_for_cone)
    {
        pdhcg_device_compute_diagonal_q_next_pdhg_primal_solution_major(
            state->current_primal_solution,
            state->pdhg_primal_solution,
            state->reflected_primal_solution,
            state->quadratic_objective_term->diagonal_objective_matrix,
            state->dual_product,
            state->objective_vector,
            state->variable_lower_bound,
            state->variable_upper_bound,
            state->num_variables,
            step_size);
    }
    else
    {
        pdhcg_device_compute_diagonal_q_next_pdhg_primal_solution(
            state->current_primal_solution,
            state->reflected_primal_solution,
            state->quadratic_objective_term->diagonal_objective_matrix,
            state->dual_product,
            state->objective_vector,
            state->variable_lower_bound,
            state->variable_upper_bound,
            state->num_variables,
            step_size);
    }
}

void primal_BB_step_size_update(pdhg_solver_state_t *state, double step_size)
{
    double inv_step_size = 1.0 / step_size;
    int inner_solver_iter = 1;
    double initial_alpha = 1.0 / inv_step_size;

    bb_step_size_t *bb = state->inner_solver->bb_step_size;
    bool precond = bb->precond_enabled;

    double *d_norm_gtg = bb->scalar_buffer;
    double *d_tmp = bb->scalar_buffer + 1;
    double *d_alpha = bb->scalar_buffer + 2;
    double *d_stMs = bb->scalar_buffer + 3;

    if (precond && bb->cached_inv_tau != inv_step_size)
    {
        pdhcg_device_refresh_inner_precond(
            bb->diag_h_static, inv_step_size, bb->m_diag, bb->m_inv, state->num_variables);
        bb->cached_inv_tau = inv_step_size;

        double sum_m = 0.0;
        DEVICE_CHECK(pdhcg_device_asum(state->blas_handle, state->num_variables, bb->m_diag, 1, &sum_m));
        pdhcg_all_reduce_scalar(state->grid_context, &sum_m, PDHCG_OP_SUM, PDHCG_SCOPE_ROW, false);
        int n_global = get_global_n(state);
        if (n_global > 0)
            bb->tol_scale = sqrt(sum_m / (double)n_global);
        else
            bb->tol_scale = 1.0;
    }

    if (precond)
        initial_alpha *= bb->tol_scale * bb->tol_scale;

    update_obj_product(state, state->current_primal_solution);
    if (precond)
    {
        pdhcg_device_primal_gradient_descent_kernel_bb_init_precond(state->dual_product,
                                                                    bb->gradient,
                                                                    bb->direction,
                                                                    state->current_primal_solution,
                                                                    state->pdhg_primal_solution,
                                                                    state->objective_vector,
                                                                    state->quadratic_objective_term->primal_obj_product,
                                                                    state->variable_lower_bound,
                                                                    state->variable_upper_bound,
                                                                    bb->m_inv,
                                                                    initial_alpha,
                                                                    state->num_variables);
    }
    else
    {
        pdhcg_device_primal_gradient_descent_kernel_bb_init(state->dual_product,
                                                            bb->gradient,
                                                            bb->direction,
                                                            state->current_primal_solution,
                                                            state->pdhg_primal_solution,
                                                            state->objective_vector,
                                                            state->quadratic_objective_term->primal_obj_product,
                                                            state->variable_lower_bound,
                                                            state->variable_upper_bound,
                                                            initial_alpha,
                                                            state->num_variables);
    }

    if (state->has_variable_cones)
    {
        project_cone_runtime(state, &state->cones, state->pdhg_primal_solution, state->cones.projection_workspace);
        pdhcg_device_vector_sub(
            bb->direction, state->pdhg_primal_solution, state->current_primal_solution, state->num_variables);
    }

    pdhcg_device_set_pointer_mode(state->blas_handle, PDHCG_POINTER_DEVICE);

    int check_frequency = 1;
    double h_norm_gtg = 0.0;

    while (inner_solver_iter < state->inner_solver->iteration_limit)
    {
        if (precond)
        {
            pdhcg_device_element_wise_mul(bb->m_diag, bb->direction, bb->Ms_buffer, state->num_variables);
            DEVICE_CHECK(
                pdhcg_device_dot(state->blas_handle, state->num_variables, bb->direction, 1, bb->Ms_buffer, 1, d_stMs));
            pdhcg_all_reduce_scalar(state->grid_context, d_stMs, PDHCG_OP_SUM, PDHCG_SCOPE_ROW, true);
            pdhcg_device_scalar_sqrt_copy(d_stMs, d_norm_gtg);
        }
        else
        {
            DEVICE_CHECK(pdhcg_device_dot(
                state->blas_handle, state->num_variables, bb->direction, 1, bb->direction, 1, d_norm_gtg));
            pdhcg_all_reduce_scalar(state->grid_context, d_norm_gtg, PDHCG_OP_SUM, PDHCG_SCOPE_ROW, true);
            pdhcg_device_scalar_sqrt_copy(d_norm_gtg, d_norm_gtg);
        }

        if (inner_solver_iter == 1 || inner_solver_iter % check_frequency == 0)
        {
            pdhcg_device_copy(&h_norm_gtg, d_norm_gtg, sizeof(double), PDHCG_COPY_DEVICE_TO_HOST);
            if (h_norm_gtg <= state->inner_solver->tol)
                break;
        }

        update_obj_product(state, state->pdhg_primal_solution);
        pdhcg_device_primal_bb_update_gradient(state->pdhg_primal_solution,
                                               state->current_primal_solution,
                                               state->objective_vector,
                                               state->dual_product,
                                               state->quadratic_objective_term->primal_obj_product,
                                               bb->gradient,
                                               state->inner_solver->primal_buffer,
                                               inv_step_size,
                                               state->num_variables);

        DEVICE_CHECK(pdhcg_device_dot(
            state->blas_handle, state->num_variables, bb->direction, 1, state->inner_solver->primal_buffer, 1, d_tmp));

        pdhcg_all_reduce_scalar(state->grid_context, d_tmp, PDHCG_OP_SUM, PDHCG_SCOPE_ROW, true);

        if (state->has_variable_cones && state->cones.bb_primal_snapshot)
        {
            DEVICE_CHECK(pdhcg_device_copy_async(state->cones.bb_primal_snapshot,
                                                 state->pdhg_primal_solution,
                                                 (size_t)state->num_variables * sizeof(double),
                                                 PDHCG_COPY_DEVICE_TO_DEVICE));
        }

        if (precond)
        {
            pdhcg_device_compute_bb_alpha_M(d_stMs, d_tmp, d_alpha);

            pdhcg_device_primal_bb_update_direction_kernel_precond(state->pdhg_primal_solution,
                                                                   bb->gradient,
                                                                   bb->direction,
                                                                   state->variable_lower_bound,
                                                                   state->variable_upper_bound,
                                                                   bb->m_inv,
                                                                   d_alpha,
                                                                   state->num_variables);
        }
        else
        {
            pdhcg_device_compute_bb_alpha_safeguard(d_norm_gtg, d_tmp, d_alpha);

            pdhcg_device_primal_bb_update_direction(state->pdhg_primal_solution,
                                                    bb->gradient,
                                                    bb->direction,
                                                    state->variable_lower_bound,
                                                    state->variable_upper_bound,
                                                    d_alpha,
                                                    state->num_variables);
        }

        if (state->has_variable_cones && state->cones.bb_primal_snapshot)
        {
            project_cone_runtime(state, &state->cones, state->pdhg_primal_solution, state->cones.projection_workspace);
            pdhcg_device_vector_sub(
                bb->direction, state->pdhg_primal_solution, state->cones.bb_primal_snapshot, state->num_variables);
        }

        inner_solver_iter++;
    }

    pdhcg_device_set_pointer_mode(state->blas_handle, PDHCG_POINTER_HOST);

    pdhcg_device_primal_bb_final(state->current_primal_solution,
                                 state->pdhg_primal_solution,
                                 state->reflected_primal_solution,
                                 state->num_variables);
    state->inner_solver->total_count += inner_solver_iter;
}

static void linearized_quadratic_primal_update(pdhg_solver_state_t *state, double step_size)
{
    double inv_step_size = 1.0 / step_size;
    /* S = L I - H cancels the quadratic prox Hessian, leaving one projected gradient step. */
    double linearization_norm = safeguarded_spectral_estimate(state->quadratic_objective_term->norm);
    double alpha = 1.0 / (linearization_norm + inv_step_size);
    update_obj_product(state, state->current_primal_solution);
    bool store_candidate = state->is_this_major_iteration || state->has_variable_cones ||
        ((state->total_count + 2) % get_print_frequency(state->total_count + 2)) == 0;
    if (store_candidate)
    {
        pdhcg_device_primal_gradient_descent_kernel_major(state->dual_product,
                                                          state->current_primal_solution,
                                                          state->reflected_primal_solution,
                                                          state->pdhg_primal_solution,
                                                          state->objective_vector,
                                                          state->quadratic_objective_term->primal_obj_product,
                                                          state->variable_lower_bound,
                                                          state->variable_upper_bound,
                                                          alpha,
                                                          state->num_variables);
    }
    else
    {
        pdhcg_device_primal_gradient_descent(state->dual_product,
                                             state->current_primal_solution,
                                             state->reflected_primal_solution,
                                             state->objective_vector,
                                             state->quadratic_objective_term->primal_obj_product,
                                             state->variable_lower_bound,
                                             state->variable_upper_bound,
                                             alpha,
                                             state->num_variables);
    }

    if (state->has_variable_cones)
    {
        project_cone_runtime(state, &state->cones, state->pdhg_primal_solution, state->cones.projection_workspace);
        recompute_cone_reflection(state);
    }
}

void pdhg_update(pdhg_solver_state_t *state)
{
    double primal_step_size = state->step_size / state->primal_weight;
    if (state->quadratic_objective_term->nonconvexity < 0)
    {
        primal_step_size = fmax(primal_step_size, -1.01 * fmin(0.0, state->quadratic_objective_term->nonconvexity));
        primal_step_size /= 100;
    }
    double dual_step_size = state->step_size * state->primal_weight;

    pdhcg_spmv_execute(state->sparse_handle,
                       state->spmv_ctx_At,
                       &HOST_ONE,
                       &HOST_ZERO,
                       state->current_dual_solution,
                       state->dual_product);

    pdhcg_all_reduce_array(
        state->grid_context, state->dual_product, state->num_variables, PDHCG_OP_SUM, PDHCG_SCOPE_COL, 0);

    if (state->use_linearized_quadratic_update)
    {
        linearized_quadratic_primal_update(state, primal_step_size);
    }
    else
    {
        switch (state->quadratic_objective_term->quad_obj_type)
        {
            case PDHCG_NON_Q:
            {
                lp_primal_update(state, primal_step_size);
                if (state->has_variable_cones)
                {
                    project_cone_runtime(
                        state, &state->cones, state->pdhg_primal_solution, state->cones.projection_workspace);
                    recompute_cone_reflection(state);
                }
                break;
            }
            case PDHCG_DIAG_Q:
            {
                if (state->cones.has_psd_cones)
                    primal_BB_step_size_update(state, primal_step_size);
                else
                {
                    diag_q_primal_update(state, primal_step_size);
                    if (state->has_variable_cones)
                        project_cone_runtime_diag_q(state, &state->cones, primal_step_size);
                }
                break;
            }
            case PDHCG_SPARSE_Q:
            case PDHCG_LOW_RANK_Q:
            case PDHCG_LOW_RANK_PLUS_SPARSE_Q:
            {
                primal_BB_step_size_update(state, primal_step_size);
                break;
            }
            default:
                fprintf(stderr, "Error: Unknown Quadratic Objective Type detected.\n");
                exit(EXIT_FAILURE);
        }
    }

    pdhcg_spmv_execute(state->sparse_handle,
                       state->spmv_ctx_A,
                       &HOST_ONE,
                       &HOST_ZERO,
                       state->reflected_primal_solution,
                       state->primal_product);

    pdhcg_all_reduce_array(
        state->grid_context, state->primal_product, state->num_constraints, PDHCG_OP_SUM, PDHCG_SCOPE_ROW, 0);

    if (state->num_constraints == 0)
        return;

    bool store_pdhg_dual =
        state->is_this_major_iteration || ((state->total_count + 2) % get_print_frequency(state->total_count + 2)) == 0;
    bool has_local_affine_cones = state->affine_cones.num_blocks > 0 || state->affine_cones.split;
    if (!has_local_affine_cones)
    {
        if (store_pdhg_dual)
        {
            pdhcg_device_compute_next_pdhg_dual_solution_major(state->current_dual_solution,
                                                               state->pdhg_dual_solution,
                                                               state->reflected_dual_solution,
                                                               state->primal_product,
                                                               state->affine_cone_offset,
                                                               state->constraint_lower_bound,
                                                               state->constraint_upper_bound,
                                                               state->num_constraints,
                                                               dual_step_size);
        }
        else
        {
            pdhcg_device_compute_next_pdhg_dual_solution(state->current_dual_solution,
                                                         state->reflected_dual_solution,
                                                         state->primal_product,
                                                         state->affine_cone_offset,
                                                         state->constraint_lower_bound,
                                                         state->constraint_upper_bound,
                                                         state->num_constraints,
                                                         dual_step_size);
        }
        return;
    }

    /* reflected_dual is scratch until the post-projection kernel on non-major iterations. */
    double *projection_point = store_pdhg_dual ? state->pdhg_dual_solution : state->reflected_dual_solution;
    pdhcg_device_prepare_constraint_dual_update(state->current_dual_solution,
                                                state->primal_product,
                                                state->affine_cone_offset,
                                                state->constraint_lower_bound,
                                                state->constraint_upper_bound,
                                                projection_point,
                                                state->num_constraints,
                                                dual_step_size);
    project_cone_runtime(state, &state->affine_cones, projection_point, state->affine_cones.projection_workspace);
    pdhcg_device_finish_constraint_dual_update(state->current_dual_solution,
                                               state->primal_product,
                                               state->affine_cone_offset,
                                               projection_point,
                                               store_pdhg_dual ? state->pdhg_dual_solution : NULL,
                                               state->reflected_dual_solution,
                                               state->num_constraints,
                                               dual_step_size);
}

void halpern_update(pdhg_solver_state_t *state, double reflection_coefficient)
{
    double weight = (double)(state->inner_count + 1) / (state->inner_count + 2);
    pdhcg_device_halpern_update(state->initial_primal_solution,
                                state->current_primal_solution,
                                state->reflected_primal_solution,
                                state->initial_dual_solution,
                                state->current_dual_solution,
                                state->reflected_dual_solution,
                                state->num_variables,
                                state->num_constraints,
                                weight,
                                reflection_coefficient);
}

void rescale_solution(pdhg_solver_state_t *state)
{
    pdhcg_device_rescale_solution(state->pdhg_primal_solution,
                                  state->pdhg_dual_solution,
                                  state->variable_rescaling,
                                  state->constraint_rescaling,
                                  state->objective_vector_rescaling,
                                  state->constraint_bound_rescaling,
                                  state->num_variables,
                                  state->num_constraints);
}

void perform_restart(pdhg_solver_state_t *state, const pdhg_parameters_t *params)
{
    pdhcg_device_compute_delta_solution(state->initial_primal_solution,
                                        state->pdhg_primal_solution,
                                        state->delta_primal_solution,
                                        state->initial_dual_solution,
                                        state->pdhg_dual_solution,
                                        state->delta_dual_solution,
                                        state->num_variables,
                                        state->num_constraints);

    double primal_dist, dual_dist;
    DEVICE_CHECK(
        pdhcg_device_nrm2(state->blas_handle, state->num_variables, state->delta_primal_solution, 1, &primal_dist));
    DEVICE_CHECK(
        pdhcg_device_nrm2(state->blas_handle, state->num_constraints, state->delta_dual_solution, 1, &dual_dist));

    double primal_dist_sq = primal_dist * primal_dist;
    pdhcg_all_reduce_scalar(state->grid_context, &primal_dist_sq, PDHCG_OP_SUM, PDHCG_SCOPE_ROW, false);
    primal_dist = sqrt(primal_dist_sq);

    double dual_dist_sq = dual_dist * dual_dist;
    pdhcg_all_reduce_scalar(state->grid_context, &dual_dist_sq, PDHCG_OP_SUM, PDHCG_SCOPE_COL, false);
    dual_dist = sqrt(dual_dist_sq);

    double ratio_infeas = state->relative_dual_residual / state->relative_primal_residual;

    if (primal_dist > 1e-16 && dual_dist > 1e-16 && primal_dist < 1e12 && dual_dist < 1e12 && ratio_infeas > 1e-8 &&
        ratio_infeas < 1e8)
    {
        double error = log(dual_dist) - log(primal_dist) - log(state->primal_weight);
        state->primal_weight_error_sum *= params->restart_params.i_smooth;
        state->primal_weight_error_sum += error;
        double delta_error = error - state->primal_weight_last_error;
        state->primal_weight *=
            exp(params->restart_params.k_p * error + params->restart_params.k_i * state->primal_weight_error_sum +
                params->restart_params.k_d * delta_error);
        state->primal_weight_last_error = error;
    }
    else
    {
        state->primal_weight = state->best_primal_weight;
        state->primal_weight_error_sum = 0.0;
        state->primal_weight_last_error = 0.0;
    }

    double primal_dual_residual_gap = fabs(log10(state->relative_dual_residual / state->relative_primal_residual));
    if (primal_dual_residual_gap < state->best_primal_dual_residual_gap)
    {
        state->best_primal_dual_residual_gap = primal_dual_residual_gap;
        state->best_primal_weight = state->primal_weight;
    }

    DEVICE_CHECK(pdhcg_device_copy(state->initial_primal_solution,
                                   state->pdhg_primal_solution,
                                   state->num_variables * sizeof(double),
                                   PDHCG_COPY_DEVICE_TO_DEVICE));
    DEVICE_CHECK(pdhcg_device_copy(state->current_primal_solution,
                                   state->pdhg_primal_solution,
                                   state->num_variables * sizeof(double),
                                   PDHCG_COPY_DEVICE_TO_DEVICE));
    DEVICE_CHECK(pdhcg_device_copy(state->initial_dual_solution,
                                   state->pdhg_dual_solution,
                                   state->num_constraints * sizeof(double),
                                   PDHCG_COPY_DEVICE_TO_DEVICE));
    DEVICE_CHECK(pdhcg_device_copy(state->current_dual_solution,
                                   state->pdhg_dual_solution,
                                   state->num_constraints * sizeof(double),
                                   PDHCG_COPY_DEVICE_TO_DEVICE));

    state->inner_count = 0;
    state->last_trial_fixed_point_error = INFINITY;
}

void initialize_step_size_and_primal_weight(pdhg_solver_state_t *state, const pdhg_parameters_t *params)
{
    bool constraint_matrix_is_zero = state->constraint_matrix->num_nonzeros == 0;
    double has_nonzero_tile = constraint_matrix_is_zero ? 0.0 : 1.0;
    pdhcg_all_reduce_scalar(state->grid_context, &has_nonzero_tile, PDHCG_OP_MAX, PDHCG_SCOPE_GLOBAL, false);
    constraint_matrix_is_zero = has_nonzero_tile == 0.0;
    if (constraint_matrix_is_zero)
    {
        state->step_size = 1.0;
    }
    else
    {
        double max_sv = estimate_maximum_singular_value(state->sparse_handle,
                                                        state->blas_handle,
                                                        state->constraint_matrix,
                                                        state->constraint_matrix_t,
                                                        params->sv_max_iter,
                                                        params->sv_tol,
                                                        state->grid_context);
        if (max_sv < 1e-12)
        {
            state->step_size = 1.0;
        }
        else
        {
            state->step_size = SPECTRAL_ESTIMATE_STEP_SAFETY / max_sv;
        }
    }

    if (params->bound_objective_rescaling)
    {
        state->primal_weight = 1.0;
    }
    else
    {
        state->primal_weight = (state->objective_vector_norm + 1.0) / (state->constraint_bound_norm + 1.0);
    }
    state->best_primal_weight = state->primal_weight;
}

void compute_fixed_point_error(pdhg_solver_state_t *state)
{
    pdhcg_device_compute_delta_solution(state->current_primal_solution,
                                        state->reflected_primal_solution,
                                        state->delta_primal_solution,
                                        state->current_dual_solution,
                                        state->reflected_dual_solution,
                                        state->delta_dual_solution,
                                        state->num_variables,
                                        state->num_constraints);

    pdhcg_spmv_execute(state->sparse_handle,
                       state->spmv_ctx_At,
                       &HOST_ONE,
                       &HOST_ZERO,
                       state->delta_dual_solution,
                       state->dual_product);

    pdhcg_all_reduce_array(
        state->grid_context, state->dual_product, state->num_variables, PDHCG_OP_SUM, PDHCG_SCOPE_COL, 0);

    double interaction, movement;

    double primal_norm = 0.0;
    double dual_norm = 0.0;
    double cross_term = 0.0;

    DEVICE_CHECK(
        pdhcg_device_nrm2(state->blas_handle, state->num_constraints, state->delta_dual_solution, 1, &dual_norm));
    DEVICE_CHECK(
        pdhcg_device_nrm2(state->blas_handle, state->num_variables, state->delta_primal_solution, 1, &primal_norm));

    double dual_norm_sq = dual_norm * dual_norm;
    pdhcg_all_reduce_scalar(state->grid_context, &dual_norm_sq, PDHCG_OP_SUM, PDHCG_SCOPE_COL, false);
    dual_norm = sqrt(dual_norm_sq);

    double primal_norm_sq = primal_norm * primal_norm;
    pdhcg_all_reduce_scalar(state->grid_context, &primal_norm_sq, PDHCG_OP_SUM, PDHCG_SCOPE_ROW, false);
    primal_norm = sqrt(primal_norm_sq);

    movement = primal_norm_sq * state->primal_weight + dual_norm_sq / state->primal_weight;

    if (state->use_linearized_quadratic_update)
    {
        /* The fixed-point metric gains eta * <dx, (L I - H) dx>. */
        update_obj_product(state, state->delta_primal_solution);
        double delta_h_delta =
            compute_xQx(state, state->delta_primal_solution, state->quadratic_objective_term->primal_obj_product);
        double linearization_norm = safeguarded_spectral_estimate(state->quadratic_objective_term->norm);
        movement += state->step_size * (linearization_norm * primal_norm_sq - delta_h_delta);
    }

    DEVICE_CHECK(pdhcg_device_dot(state->blas_handle,
                                  state->num_variables,
                                  state->dual_product,
                                  1,
                                  state->delta_primal_solution,
                                  1,
                                  &cross_term));

    pdhcg_all_reduce_scalar(state->grid_context, &cross_term, PDHCG_OP_SUM, PDHCG_SCOPE_ROW, false);

    interaction = 2 * state->step_size * cross_term;

    state->fixed_point_error = sqrt(fmax(0.0, movement + interaction));
    if (state->problem_type == CONVEX_QP && state->inner_solver->bb_step_size)
    {
        state->inner_solver->tol =
            fmin(state->inner_solver->tol,
                 fmax(0.0005 * primal_norm / state->step_size * state->primal_weight, state->inner_solver->min_tol));
    }
}

void compute_residual(pdhg_solver_state_t *state, norm_type_t optimality_norm)
{
    double linear_absolute_primal_residual = 0.0;
    double power_cone_absolute_violation = 0.0;
    double power_cone_relative_violation = 0.0;
    double affine_dual_membership_norm = 0.0;
    double affine_complementarity_norm = 0.0;
    bool has_affine_cones = has_affine_cone_constraints(state);
    pdhcg_spmv_execute(state->sparse_handle,
                       state->spmv_ctx_A,
                       &HOST_ONE,
                       &HOST_ZERO,
                       state->pdhg_primal_solution,
                       state->primal_product);

    pdhcg_all_reduce_array(
        state->grid_context, state->primal_product, state->num_constraints, PDHCG_OP_SUM, PDHCG_SCOPE_ROW, 0);

    pdhcg_spmv_execute(state->sparse_handle,
                       state->spmv_ctx_At,
                       &HOST_ONE,
                       &HOST_ZERO,
                       state->pdhg_dual_solution,
                       state->dual_product);

    pdhcg_all_reduce_array(
        state->grid_context, state->dual_product, state->num_variables, PDHCG_OP_SUM, PDHCG_SCOPE_COL, 0);

    update_obj_product(state, state->pdhg_primal_solution);
    update_cone_effective_objective_gradient(state);

    if (state->problem_type == LP)
    {
        pdhcg_device_compute_lp_residual(state->primal_residual,
                                         state->primal_product,
                                         state->affine_cone_offset,
                                         state->constraint_lower_bound,
                                         state->constraint_upper_bound,
                                         state->pdhg_dual_solution,
                                         state->dual_residual,
                                         state->dual_product,
                                         state->dual_slack,
                                         state->objective_vector,
                                         state->constraint_rescaling,
                                         state->variable_rescaling,
                                         state->delta_dual_solution,
                                         state->primal_slack,
                                         state->constraint_lower_bound_finite_val,
                                         state->constraint_upper_bound_finite_val,
                                         has_affine_cones,
                                         state->num_constraints,
                                         state->num_variables);

        if (state->has_variable_cones)
        {
            const double *effective_obj = cone_dual_residual_effective_obj(state);
            compute_cone_dual_residual(state, effective_obj);
            augment_conic_projected_gradient_residual(state, effective_obj);
        }
    }
    else if (state->problem_type == CONVEX_QP)
    {
        pdhcg_device_compute_qp_residual(state->primal_residual,
                                         state->primal_product,
                                         state->affine_cone_offset,
                                         state->quadratic_objective_term->primal_obj_product,
                                         state->pdhg_primal_solution,
                                         state->constraint_lower_bound,
                                         state->constraint_upper_bound,
                                         state->variable_lower_bound,
                                         state->variable_upper_bound,
                                         state->pdhg_dual_solution,
                                         state->dual_residual,
                                         state->dual_product,
                                         state->dual_slack,
                                         state->objective_vector,
                                         state->constraint_rescaling,
                                         state->variable_rescaling,
                                         state->delta_dual_solution,
                                         state->primal_slack,
                                         state->constraint_lower_bound_finite_val,
                                         state->constraint_upper_bound_finite_val,
                                         state->step_size / state->primal_weight,
                                         has_affine_cones,
                                         state->num_constraints,
                                         state->num_variables);

        if (state->has_variable_cones)
        {
            const double *effective_obj = cone_dual_residual_effective_obj(state);
            compute_cone_dual_residual(state, effective_obj);
            augment_conic_projected_gradient_residual(state, effective_obj);
        }
    }
    if (state->affine_cones.num_blocks > 0 || state->affine_cones.split)
    {
        project_cone_runtime(
            state, &state->affine_cones, state->primal_residual, state->affine_cones.residual_workspace);
    }
    compute_affine_cone_residuals(state, optimality_norm, &affine_dual_membership_norm, &affine_complementarity_norm);
    if (optimality_norm == NORM_TYPE_L_INF)
    {
        state->absolute_primal_residual =
            get_vector_inf_norm(state->blas_handle, state->num_constraints, state->primal_residual);
        pdhcg_all_reduce_scalar(
            state->grid_context, &state->absolute_primal_residual, PDHCG_OP_MAX, PDHCG_SCOPE_COL, false);
    }
    else
    {
        DEVICE_CHECK(pdhcg_device_nrm2(
            state->blas_handle, state->num_constraints, state->primal_residual, 1, &state->absolute_primal_residual));
        state->absolute_primal_residual *= state->absolute_primal_residual;
        pdhcg_all_reduce_scalar(
            state->grid_context, &state->absolute_primal_residual, PDHCG_OP_SUM, PDHCG_SCOPE_COL, false);
        state->absolute_primal_residual = sqrt(state->absolute_primal_residual);
    }
    state->absolute_primal_residual /= state->constraint_bound_rescaling;
    linear_absolute_primal_residual = state->absolute_primal_residual;
    if (state->has_variable_cones)
    {
        compute_power_cone_primal_violation(
            state, optimality_norm, &power_cone_absolute_violation, &power_cone_relative_violation);
        if (optimality_norm == NORM_TYPE_L_INF)
            state->absolute_primal_residual = fmax(state->absolute_primal_residual, power_cone_absolute_violation);
        else
            state->absolute_primal_residual = hypot(state->absolute_primal_residual, power_cone_absolute_violation);
    }

    if (optimality_norm == NORM_TYPE_L_INF)
    {
        state->absolute_dual_residual =
            get_vector_inf_norm(state->blas_handle, state->num_variables, state->dual_residual);
        state->absolute_dual_residual =
            fmax(state->absolute_dual_residual, compute_cone_complementarity_norm(state, optimality_norm));
        pdhcg_all_reduce_scalar(
            state->grid_context, &state->absolute_dual_residual, PDHCG_OP_MAX, PDHCG_SCOPE_ROW, false);
        state->absolute_dual_residual = fmax(state->absolute_dual_residual, affine_dual_membership_norm);
        state->absolute_dual_residual = fmax(state->absolute_dual_residual, affine_complementarity_norm);
    }
    else
    {
        DEVICE_CHECK(pdhcg_device_nrm2(
            state->blas_handle, state->num_variables, state->dual_residual, 1, &state->absolute_dual_residual));
        state->absolute_dual_residual *= state->absolute_dual_residual;
        double complementarity_norm = compute_cone_complementarity_norm(state, optimality_norm);
        state->absolute_dual_residual += complementarity_norm * complementarity_norm;
        pdhcg_all_reduce_scalar(
            state->grid_context, &state->absolute_dual_residual, PDHCG_OP_SUM, PDHCG_SCOPE_ROW, false);
        state->absolute_dual_residual += affine_dual_membership_norm * affine_dual_membership_norm;
        state->absolute_dual_residual += affine_complementarity_norm * affine_complementarity_norm;
        state->absolute_dual_residual = sqrt(state->absolute_dual_residual);
    }
    state->absolute_dual_residual /= state->objective_vector_rescaling;

    double half_xQx =
        0.5 * compute_xQx(state, state->pdhg_primal_solution, state->quadratic_objective_term->primal_obj_product);

    DEVICE_CHECK(pdhcg_device_dot(state->blas_handle,
                                  state->num_variables,
                                  state->objective_vector,
                                  1,
                                  state->pdhg_primal_solution,
                                  1,
                                  &state->primal_objective_value));

    pdhcg_all_reduce_scalar(state->grid_context, &state->primal_objective_value, PDHCG_OP_SUM, PDHCG_SCOPE_ROW, false);

    state->primal_objective_value = (state->primal_objective_value + half_xQx) /
            (state->constraint_bound_rescaling * state->objective_vector_rescaling) +
        state->objective_constant;

    if (state->has_variable_cones)
    {
        const double *effective_obj = cone_dual_residual_effective_obj(state);
        set_cone_dual_slack(state, effective_obj);
    }

    double base_dual_objective;
    DEVICE_CHECK(pdhcg_device_dot(state->blas_handle,
                                  state->num_variables,
                                  state->dual_slack,
                                  1,
                                  state->pdhg_primal_solution,
                                  1,
                                  &base_dual_objective));

    pdhcg_all_reduce_scalar(state->grid_context, &base_dual_objective, PDHCG_OP_SUM, PDHCG_SCOPE_ROW, false);

    double dual_slack_sum =
        get_vector_sum(state->blas_handle, state->num_constraints, state->ones_dual, state->primal_slack);
    pdhcg_all_reduce_scalar(state->grid_context, &dual_slack_sum, PDHCG_OP_SUM, PDHCG_SCOPE_COL, false);

    state->dual_objective_value = (base_dual_objective + dual_slack_sum - half_xQx) /
            (state->constraint_bound_rescaling * state->objective_vector_rescaling) +
        state->objective_constant;

    double relative_primal_dominator = 1.0 + state->constraint_bound_norm;
    state->relative_primal_residual = linear_absolute_primal_residual / relative_primal_dominator;
    if (optimality_norm == NORM_TYPE_L_INF)
        state->relative_primal_residual = fmax(state->relative_primal_residual, power_cone_relative_violation);
    else
        state->relative_primal_residual = hypot(state->relative_primal_residual, power_cone_relative_violation);

    double relative_dual_dominator;
    if (state->problem_type == LP)
    {
        relative_dual_dominator = 1.0 + state->objective_vector_norm;
    }
    else
    {
        pdhcg_device_recover_primal_obj_dual_product(state->dual_product,
                                                     state->quadratic_objective_term->primal_obj_product,
                                                     state->variable_rescaling,
                                                     state->num_variables);
        double qx_norm;
        if (optimality_norm == NORM_TYPE_L_INF)
        {
            qx_norm = get_vector_inf_norm(
                state->blas_handle, state->num_variables, state->quadratic_objective_term->primal_obj_product);
            pdhcg_all_reduce_scalar(state->grid_context, &qx_norm, PDHCG_OP_MAX, PDHCG_SCOPE_ROW, false);
        }
        else
        {
            DEVICE_CHECK(pdhcg_device_nrm2(state->blas_handle,
                                           state->num_variables,
                                           state->quadratic_objective_term->primal_obj_product,
                                           1,
                                           &qx_norm));
            qx_norm *= qx_norm;
            pdhcg_all_reduce_scalar(state->grid_context, &qx_norm, PDHCG_OP_SUM, PDHCG_SCOPE_ROW, false);
            qx_norm = sqrt(qx_norm);
        }
        double Ay_norm;
        if (optimality_norm == NORM_TYPE_L_INF)
        {
            Ay_norm = get_vector_inf_norm(state->blas_handle, state->num_variables, state->dual_product);
            pdhcg_all_reduce_scalar(state->grid_context, &Ay_norm, PDHCG_OP_MAX, PDHCG_SCOPE_ROW, false);
        }
        else
        {
            DEVICE_CHECK(pdhcg_device_nrm2(state->blas_handle, state->num_variables, state->dual_product, 1, &Ay_norm));
            Ay_norm *= Ay_norm;
            pdhcg_all_reduce_scalar(state->grid_context, &Ay_norm, PDHCG_OP_SUM, PDHCG_SCOPE_ROW, false);
            Ay_norm = sqrt(Ay_norm);
        }
        relative_dual_dominator = 1.0 +
            fmax(state->objective_vector_norm,
                 fmax(qx_norm / state->objective_vector_rescaling, Ay_norm / state->objective_vector_rescaling));
    }
    state->relative_dual_residual = state->absolute_dual_residual / relative_dual_dominator;

    state->objective_gap = fabs(state->primal_objective_value - state->dual_objective_value);

    state->relative_objective_gap =
        state->objective_gap / (1.0 + fabs(state->primal_objective_value) + fabs(state->dual_objective_value));
}

pdhcg_result_t *create_result_from_state(pdhg_solver_state_t *state, const qp_problem_t *original_problem)
{
    pdhcg_result_t *results = (pdhcg_result_t *)safe_calloc(1, sizeof(pdhcg_result_t));

    pdhcg_spmv_execute(state->sparse_handle,
                       state->spmv_ctx_At,
                       &HOST_ONE,
                       &HOST_ZERO,
                       state->pdhg_dual_solution,
                       state->dual_product);

    update_obj_product(state, state->pdhg_primal_solution);

    pdhcg_device_compute_and_rescale_reduced_cost_qp(state->dual_slack,
                                                     state->objective_vector,
                                                     state->quadratic_objective_term->primal_obj_product,
                                                     state->dual_product,
                                                     state->variable_rescaling,
                                                     state->objective_vector_rescaling,
                                                     state->constraint_bound_rescaling,
                                                     state->variable_lower_bound,
                                                     state->variable_upper_bound,
                                                     state->num_variables);

    rescale_solution(state);

    results->primal_solution = (double *)safe_malloc(state->num_variables * sizeof(double));
    results->dual_solution = (double *)safe_malloc(state->num_constraints * sizeof(double));
    results->reduced_cost = (double *)safe_malloc(state->num_variables * sizeof(double));

    DEVICE_CHECK(pdhcg_device_copy(results->primal_solution,
                                   state->pdhg_primal_solution,
                                   state->num_variables * sizeof(double),
                                   PDHCG_COPY_DEVICE_TO_HOST));
    DEVICE_CHECK(pdhcg_device_copy(results->dual_solution,
                                   state->pdhg_dual_solution,
                                   state->num_constraints * sizeof(double),
                                   PDHCG_COPY_DEVICE_TO_HOST));
    DEVICE_CHECK(pdhcg_device_copy(
        results->reduced_cost, state->dual_slack, state->num_variables * sizeof(double), PDHCG_COPY_DEVICE_TO_HOST));

    results->num_variables = original_problem->num_variables;
    results->num_constraints = original_problem->num_constraints;
    results->num_nonzeros = original_problem->constraint_matrix_num_nonzeros;
    results->total_count = state->total_count;
    results->rescaling_time_sec = state->rescaling_time_sec;
    results->cumulative_time_sec = state->cumulative_time_sec;
    results->relative_primal_residual = state->relative_primal_residual;
    results->relative_dual_residual = state->relative_dual_residual;
    results->absolute_primal_residual = state->absolute_primal_residual;
    results->absolute_dual_residual = state->absolute_dual_residual;
    results->primal_objective_value = state->primal_objective_value;
    results->dual_objective_value = state->dual_objective_value;
    results->objective_gap = state->objective_gap;
    results->relative_objective_gap = state->relative_objective_gap;
    results->max_primal_ray_infeasibility = state->max_primal_ray_infeasibility;
    results->max_dual_ray_infeasibility = state->max_dual_ray_infeasibility;
    results->primal_ray_linear_objective = state->primal_ray_linear_objective;
    results->dual_ray_objective = state->dual_ray_objective;
    results->termination_reason = state->termination_reason;
    results->feasibility_polishing_time = state->feasibility_polishing_time;
    results->feasibility_iteration = state->feasibility_iteration;
    results->total_inner_count = state->inner_solver->total_count;
    return results;
}

static bool spectral_error_within_tolerance(double error, double estimate, double tolerance)
{
    return fabs(error) <= tolerance * fmax(1.0, fabs(estimate));
}

double estimate_quadratic_objective_norm(pdhcg_device_sparse_t sparse_handle,
                                         pdhcg_device_blas_t blas_handle,
                                         quadratic_objective_term_t *quadratic_objective,
                                         int num_variables,
                                         int max_iterations,
                                         double tolerance,
                                         grid_context_t *grid_context)
{
    int n = num_variables;
    if (n <= 0)
        return 0.0;

    double *vector = NULL;
    double *image_vector = NULL;
    double *next_vector = NULL;
    DEVICE_CHECK(pdhcg_device_allocate((void **)&vector, (size_t)n * sizeof(double)));
    DEVICE_CHECK(pdhcg_device_allocate((void **)&image_vector, (size_t)n * sizeof(double)));
    DEVICE_CHECK(pdhcg_device_allocate((void **)&next_vector, (size_t)n * sizeof(double)));

    double *host_vector = (double *)safe_malloc((size_t)n * sizeof(double));
    unsigned int seed = 1234U + (unsigned int)get_n_start(grid_context);
    for (int i = 0; i < n; ++i)
        host_vector[i] = 2.0 * (double)rand_r(&seed) / RAND_MAX - 1.0;
    DEVICE_CHECK(pdhcg_device_copy(vector, host_vector, (size_t)n * sizeof(double), PDHCG_COPY_HOST_TO_DEVICE));
    free(host_vector);

    double estimate = 0.0;
    for (int iteration = 0; iteration < max_iterations; ++iteration)
    {
        double local_norm = 0.0;
        DEVICE_CHECK(pdhcg_device_nrm2(blas_handle, n, vector, 1, &local_norm));
        double norm_squared = local_norm * local_norm;
        pdhcg_all_reduce_scalar(grid_context, &norm_squared, PDHCG_OP_SUM, PDHCG_SCOPE_ROW, false);
        double norm = sqrt(norm_squared);
        if (!(norm > 0.0) || !isfinite(norm))
            break;

        double inverse_norm = 1.0 / norm;
        DEVICE_CHECK(pdhcg_device_scal(blas_handle, n, &inverse_norm, vector, 1));
        compute_quadratic_objective_product(sparse_handle, blas_handle, quadratic_objective, vector, n, grid_context);
        DEVICE_CHECK(pdhcg_device_copy(image_vector,
                                       quadratic_objective->primal_obj_product,
                                       (size_t)n * sizeof(double),
                                       PDHCG_COPY_DEVICE_TO_DEVICE));

        double local_image_norm = 0.0;
        DEVICE_CHECK(pdhcg_device_nrm2(blas_handle, n, image_vector, 1, &local_image_norm));
        double image_norm_squared = local_image_norm * local_image_norm;
        pdhcg_all_reduce_scalar(grid_context, &image_norm_squared, PDHCG_OP_SUM, PDHCG_SCOPE_ROW, false);
        estimate = sqrt(image_norm_squared);
        if (!(estimate > 0.0) || !isfinite(estimate))
            break;

        compute_quadratic_objective_product(
            sparse_handle, blas_handle, quadratic_objective, image_vector, n, grid_context);
        DEVICE_CHECK(pdhcg_device_copy(next_vector,
                                       quadratic_objective->primal_obj_product,
                                       (size_t)n * sizeof(double),
                                       PDHCG_COPY_DEVICE_TO_DEVICE));

        double eigenvalue = estimate * estimate;
        double negative_eigenvalue = -eigenvalue;
        DEVICE_CHECK(pdhcg_device_axpy(blas_handle, n, &negative_eigenvalue, vector, 1, next_vector, 1));
        double local_residual_norm = 0.0;
        DEVICE_CHECK(pdhcg_device_nrm2(blas_handle, n, next_vector, 1, &local_residual_norm));
        double residual_norm_squared = local_residual_norm * local_residual_norm;
        pdhcg_all_reduce_scalar(grid_context, &residual_norm_squared, PDHCG_OP_SUM, PDHCG_SCOPE_ROW, false);
        if (spectral_error_within_tolerance(sqrt(residual_norm_squared), eigenvalue, tolerance))
            break;

        DEVICE_CHECK(pdhcg_device_axpy(blas_handle, n, &eigenvalue, vector, 1, next_vector, 1));
        double *swap = vector;
        vector = next_vector;
        next_vector = swap;
    }

    DEVICE_CHECK(pdhcg_device_free(vector));
    DEVICE_CHECK(pdhcg_device_free(image_vector));
    DEVICE_CHECK(pdhcg_device_free(next_vector));
    return estimate;
}

double estimate_quadratic_objective_minimum_eigenvalue(pdhcg_device_sparse_t sparse_handle,
                                                       pdhcg_device_blas_t blas_handle,
                                                       quadratic_objective_term_t *quadratic_objective,
                                                       int num_variables,
                                                       double spectral_norm,
                                                       int max_iterations,
                                                       double tolerance,
                                                       grid_context_t *grid_context)
{
    int n = num_variables;
    if (n <= 0)
        return 0.0;

    double *vector = NULL;
    double *shifted_vector = NULL;
    DEVICE_CHECK(pdhcg_device_allocate((void **)&vector, (size_t)n * sizeof(double)));
    DEVICE_CHECK(pdhcg_device_allocate((void **)&shifted_vector, (size_t)n * sizeof(double)));

    double *host_vector = (double *)safe_malloc((size_t)n * sizeof(double));
    unsigned int seed = 1234U + (unsigned int)get_n_start(grid_context);
    for (int i = 0; i < n; ++i)
        host_vector[i] = 2.0 * (double)rand_r(&seed) / RAND_MAX - 1.0;
    DEVICE_CHECK(pdhcg_device_copy(vector, host_vector, (size_t)n * sizeof(double), PDHCG_COPY_HOST_TO_DEVICE));
    free(host_vector);

    double shift = safeguarded_spectral_estimate(spectral_norm);
    double mu = 0.0;
    for (int iteration = 0; iteration < max_iterations; ++iteration)
    {
        double local_norm = 0.0;
        DEVICE_CHECK(pdhcg_device_nrm2(blas_handle, n, vector, 1, &local_norm));
        double norm_squared = local_norm * local_norm;
        pdhcg_all_reduce_scalar(grid_context, &norm_squared, PDHCG_OP_SUM, PDHCG_SCOPE_ROW, false);
        double norm = sqrt(norm_squared);
        if (!(norm > 0.0) || !isfinite(norm))
            break;

        double inverse_norm = 1.0 / norm;
        DEVICE_CHECK(pdhcg_device_scal(blas_handle, n, &inverse_norm, vector, 1));
        compute_quadratic_objective_product(sparse_handle, blas_handle, quadratic_objective, vector, n, grid_context);
        DEVICE_CHECK(pdhcg_device_copy(shifted_vector,
                                       quadratic_objective->primal_obj_product,
                                       (size_t)n * sizeof(double),
                                       PDHCG_COPY_DEVICE_TO_DEVICE));

        double negative_one = -1.0;
        DEVICE_CHECK(pdhcg_device_scal(blas_handle, n, &negative_one, shifted_vector, 1));
        DEVICE_CHECK(pdhcg_device_axpy(blas_handle, n, &shift, vector, 1, shifted_vector, 1));

        DEVICE_CHECK(pdhcg_device_dot(blas_handle, n, vector, 1, shifted_vector, 1, &mu));
        pdhcg_all_reduce_scalar(grid_context, &mu, PDHCG_OP_SUM, PDHCG_SCOPE_ROW, false);

        double negative_mu = -mu;
        DEVICE_CHECK(pdhcg_device_axpy(blas_handle, n, &negative_mu, vector, 1, shifted_vector, 1));
        double local_residual_norm = 0.0;
        DEVICE_CHECK(pdhcg_device_nrm2(blas_handle, n, shifted_vector, 1, &local_residual_norm));
        double residual_norm_squared = local_residual_norm * local_residual_norm;
        pdhcg_all_reduce_scalar(grid_context, &residual_norm_squared, PDHCG_OP_SUM, PDHCG_SCOPE_ROW, false);
        if (spectral_error_within_tolerance(sqrt(residual_norm_squared), mu, tolerance))
            break;

        DEVICE_CHECK(pdhcg_device_axpy(blas_handle, n, &mu, vector, 1, shifted_vector, 1));
        double *swap = vector;
        vector = shifted_vector;
        shifted_vector = swap;
    }

    DEVICE_CHECK(pdhcg_device_free(vector));
    DEVICE_CHECK(pdhcg_device_free(shifted_vector));
    return shift - mu;
}

double estimate_maximum_singular_value(pdhcg_device_sparse_t sparse_handle,
                                       pdhcg_device_blas_t blas_handle,
                                       const device_sparse_matrix_csr_t *A,
                                       const device_sparse_matrix_csr_t *AT,
                                       int max_iterations,
                                       double tolerance,
                                       struct grid_context_s *ctx)
{
    int m = A->num_rows;
    int n = A->num_cols;

    int row_coord = pdhcg_get_grid_row_coord(ctx);

    int safe_m = m > 0 ? m : 1;
    int safe_n = n > 0 ? n : 1;
    double *eigenvector_d, *next_eigenvector_d, *dual_product_d;

    DEVICE_CHECK(pdhcg_device_allocate((void **)&eigenvector_d, safe_m * sizeof(double)));
    DEVICE_CHECK(pdhcg_device_allocate((void **)&next_eigenvector_d, safe_m * sizeof(double)));
    DEVICE_CHECK(pdhcg_device_allocate((void **)&dual_product_d, safe_n * sizeof(double)));

    double *eigenvector_h = (double *)safe_malloc(safe_m * sizeof(double));
    unsigned int seed = 1234 + row_coord;
    for (int i = 0; i < safe_m; ++i)
    {
        eigenvector_h[i] = ((double)rand_r(&seed) / (double)RAND_MAX) * 2.0 - 1.0;
    }
    if (m > 0)
        DEVICE_CHECK(pdhcg_device_copy(eigenvector_d, eigenvector_h, m * sizeof(double), PDHCG_COPY_HOST_TO_DEVICE));
    free(eigenvector_h);

    double sigma_max_sq = 1.0;

    pdhcg_device_vector_t vecEigen, vecNextEigen, vecDual;
    DEVICE_CHECK(pdhcg_device_vector_create(&vecEigen, m, eigenvector_d));
    DEVICE_CHECK(pdhcg_device_vector_create(&vecNextEigen, m, next_eigenvector_d));
    DEVICE_CHECK(pdhcg_device_vector_create(&vecDual, n, dual_product_d));

    pdhcg_spmv_ctx_t *ctx_A = pdhcg_spmv_ctx_create(
        sparse_handle, m, n, A->num_nonzeros, A->row_ptr, A->col_ind, A->val, vecDual, vecNextEigen);
    pdhcg_spmv_ctx_t *ctx_At = pdhcg_spmv_ctx_create(
        sparse_handle, n, m, AT->num_nonzeros, AT->row_ptr, AT->col_ind, AT->val, vecEigen, vecDual);

    double local_norm = 0.0;
    if (m > 0)
        DEVICE_CHECK(pdhcg_device_nrm2(blas_handle, m, eigenvector_d, 1, &local_norm));

    double norm_sq = local_norm * local_norm;
    pdhcg_all_reduce_scalar(ctx, &norm_sq, PDHCG_OP_SUM, PDHCG_SCOPE_COL, false);

    double inv_norm = 1.0 / sqrt(norm_sq);
    if (m > 0)
        DEVICE_CHECK(pdhcg_device_scal(blas_handle, m, &inv_norm, eigenvector_d, 1));

    for (int i = 0; i < max_iterations; ++i)
    {
        pdhcg_spmv_execute(sparse_handle, ctx_At, &HOST_ONE, &HOST_ZERO, eigenvector_d, dual_product_d);
        pdhcg_all_reduce_array(ctx, dual_product_d, n, PDHCG_OP_SUM, PDHCG_SCOPE_COL, 0);

        pdhcg_spmv_execute(sparse_handle, ctx_A, &HOST_ONE, &HOST_ZERO, dual_product_d, next_eigenvector_d);
        pdhcg_all_reduce_array(ctx, next_eigenvector_d, m, PDHCG_OP_SUM, PDHCG_SCOPE_ROW, 0);

        double local_dot = 0.0;
        if (m > 0)
            DEVICE_CHECK(pdhcg_device_dot(blas_handle, m, next_eigenvector_d, 1, eigenvector_d, 1, &local_dot));

        pdhcg_all_reduce_scalar(ctx, &local_dot, PDHCG_OP_SUM, PDHCG_SCOPE_COL, false);
        sigma_max_sq = local_dot;

        double neg_sigma_sq = -sigma_max_sq;
        if (m > 0)
            DEVICE_CHECK(pdhcg_device_axpy(blas_handle, m, &neg_sigma_sq, eigenvector_d, 1, next_eigenvector_d, 1));

        double local_res_norm = 0.0;
        if (m > 0)
            DEVICE_CHECK(pdhcg_device_nrm2(blas_handle, m, next_eigenvector_d, 1, &local_res_norm));

        double res_sq = local_res_norm * local_res_norm;
        pdhcg_all_reduce_scalar(ctx, &res_sq, PDHCG_OP_SUM, PDHCG_SCOPE_COL, false);

        double residual_norm = sqrt(res_sq);
        if (spectral_error_within_tolerance(residual_norm, sigma_max_sq, tolerance))
            break;

        if (m > 0)
            DEVICE_CHECK(pdhcg_device_axpy(blas_handle, m, &sigma_max_sq, eigenvector_d, 1, next_eigenvector_d, 1));

        local_norm = 0.0;
        if (m > 0)
            DEVICE_CHECK(pdhcg_device_nrm2(blas_handle, m, next_eigenvector_d, 1, &local_norm));

        norm_sq = local_norm * local_norm;
        pdhcg_all_reduce_scalar(ctx, &norm_sq, PDHCG_OP_SUM, PDHCG_SCOPE_COL, false);

        inv_norm = 1.0 / sqrt(norm_sq);
        if (m > 0)
            DEVICE_CHECK(pdhcg_device_scal(blas_handle, m, &inv_norm, next_eigenvector_d, 1));

        double *tmp = eigenvector_d;
        eigenvector_d = next_eigenvector_d;
        next_eigenvector_d = tmp;

        DEVICE_CHECK(pdhcg_device_vector_set_values(vecEigen, eigenvector_d));
        DEVICE_CHECK(pdhcg_device_vector_set_values(vecNextEigen, next_eigenvector_d));
    }

    pdhcg_spmv_ctx_destroy(ctx_A);
    pdhcg_spmv_ctx_destroy(ctx_At);
    DEVICE_CHECK(pdhcg_device_vector_destroy(vecEigen));
    DEVICE_CHECK(pdhcg_device_vector_destroy(vecNextEigen));
    DEVICE_CHECK(pdhcg_device_vector_destroy(vecDual));
    DEVICE_CHECK(pdhcg_device_free(eigenvector_d));
    DEVICE_CHECK(pdhcg_device_free(next_eigenvector_d));
    DEVICE_CHECK(pdhcg_device_free(dual_product_d));

    return sqrt(sigma_max_sq);
}
