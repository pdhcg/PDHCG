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

#include "solver_state.h"
#include "cone_dispatch.h"
#include "cone_utils.h"
#include "device_cones.h"
#include "device_kernels.h"
#include "device_scaling.h"
#include "distributed_conic.h"
#include "internal_types.h"
#include "pdhcg.h"
#include "pdhcg_psd_cone.h"
#include "pdhg_core_op.h"
#include "preprocess.h"
#include "solver.h"
#include "spmv_backend.h"
#include "utils.h"
#ifdef PDHCG_COMPILE_DISTRIBUTED
#include "distributed_types.h"
#endif
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

int get_global_n(pdhg_solver_state_t *state)
{
    int global_n = pdhcg_get_global_num_variables(state->grid_context);
    return global_n > 0 ? global_n : state->num_variables;
}

int get_n_start(grid_context_t *ctx)
{
    return pdhcg_get_variable_start(ctx);
}

static void initialize_sparse_component_obj(pdhg_solver_state_t *state, const processed_qp_problem_t *problem)
{
    state->quadratic_objective_term->objective_sparse_matrix =
        (device_sparse_matrix_csr_t *)safe_malloc(sizeof(device_sparse_matrix_csr_t));

    int q_rows = get_global_n(state);

    memset(state->quadratic_objective_term->objective_sparse_matrix, 0, sizeof(device_sparse_matrix_csr_t));
    state->quadratic_objective_term->objective_sparse_matrix->num_rows = q_rows;
    state->quadratic_objective_term->objective_sparse_matrix->num_cols = problem->num_variables;
    state->quadratic_objective_term->objective_sparse_matrix->num_nonzeros =
        problem->objective_sparse_matrix_num_nonzeros;

    ALLOC_AND_COPY_DEVICE_CSR(state->quadratic_objective_term->objective_sparse_matrix,
                              problem->objective_sparse_matrix,
                              q_rows,
                              problem->objective_sparse_matrix_num_nonzeros);
}

static void initialize_lowrank_component_obj(pdhg_solver_state_t *state, processed_qp_problem_t *problem)
{
    state->quadratic_objective_term->num_rank_lowrank_obj = problem->num_rank_lowrank_obj;
    ALLOC_ZERO(state->quadratic_objective_term->Rx_product, (size_t)problem->num_rank_lowrank_obj * sizeof(double));

    DEVICE_CHECK(pdhcg_device_vector_create(&state->quadratic_objective_term->vec_Rx_prod,
                                            state->quadratic_objective_term->num_rank_lowrank_obj,
                                            state->quadratic_objective_term->Rx_product));

    state->quadratic_objective_term->objective_lowrank_matrix =
        (device_sparse_matrix_csr_t *)safe_malloc(sizeof(device_sparse_matrix_csr_t));

    memset(state->quadratic_objective_term->objective_lowrank_matrix, 0, sizeof(device_sparse_matrix_csr_t));
    state->quadratic_objective_term->objective_lowrank_matrix->num_rows = problem->num_rank_lowrank_obj;
    state->quadratic_objective_term->objective_lowrank_matrix->num_cols = problem->num_variables;
    state->quadratic_objective_term->objective_lowrank_matrix->num_nonzeros =
        problem->objective_lowrank_matrix_num_nonzeros;

    ALLOC_AND_COPY_DEVICE_CSR(state->quadratic_objective_term->objective_lowrank_matrix,
                              problem->objective_lowrank_matrix,
                              problem->num_rank_lowrank_obj,
                              problem->objective_lowrank_matrix_num_nonzeros);

    state->quadratic_objective_term->objective_lowrank_matrix_t =
        (device_sparse_matrix_csr_t *)safe_malloc(sizeof(device_sparse_matrix_csr_t));

    memset(state->quadratic_objective_term->objective_lowrank_matrix_t, 0, sizeof(device_sparse_matrix_csr_t));
    state->quadratic_objective_term->objective_lowrank_matrix_t->num_rows = problem->num_variables;
    state->quadratic_objective_term->objective_lowrank_matrix_t->num_cols = problem->num_rank_lowrank_obj;
    state->quadratic_objective_term->objective_lowrank_matrix_t->num_nonzeros =
        problem->objective_lowrank_matrix_num_nonzeros;

    DEVICE_CHECK(pdhcg_device_allocate((void **)&state->quadratic_objective_term->objective_lowrank_matrix_t->row_ptr,
                                       ((size_t)problem->num_variables + 1) * sizeof(int)));
    DEVICE_CHECK(pdhcg_device_allocate((void **)&state->quadratic_objective_term->objective_lowrank_matrix_t->col_ind,
                                       (size_t)problem->objective_lowrank_matrix_num_nonzeros * sizeof(int)));
    DEVICE_CHECK(pdhcg_device_allocate((void **)&state->quadratic_objective_term->objective_lowrank_matrix_t->val,
                                       (size_t)problem->objective_lowrank_matrix_num_nonzeros * sizeof(double)));

    DEVICE_CHECK(pdhcg_device_csr_transpose(state->sparse_handle,
                                            state->quadratic_objective_term->objective_lowrank_matrix->num_rows,
                                            state->quadratic_objective_term->objective_lowrank_matrix->num_cols,
                                            state->quadratic_objective_term->objective_lowrank_matrix->num_nonzeros,
                                            state->quadratic_objective_term->objective_lowrank_matrix->val,
                                            state->quadratic_objective_term->objective_lowrank_matrix->row_ptr,
                                            state->quadratic_objective_term->objective_lowrank_matrix->col_ind,
                                            state->quadratic_objective_term->objective_lowrank_matrix_t->val,
                                            state->quadratic_objective_term->objective_lowrank_matrix_t->row_ptr,
                                            state->quadratic_objective_term->objective_lowrank_matrix_t->col_ind));

    state->quadratic_objective_term->lowrank_middle_type = (int)problem->objective_lowrank_middle_kind;
    state->quadratic_objective_term->d_middle_diag = NULL;
    state->quadratic_objective_term->d_middle_dense = NULL;
    state->quadratic_objective_term->Rx_buffer = NULL;

    int rank = problem->num_rank_lowrank_obj;
    if (problem->objective_lowrank_middle_kind == PDHCG_D_DIAG && rank > 0)
    {
        state->quadratic_objective_term->d_middle_diag = problem->objective_lowrank_middle_diag;
        problem->objective_lowrank_middle_diag = NULL;
    }
    else if (problem->objective_lowrank_middle_kind == PDHCG_D_DENSE && rank > 0)
    {
        state->quadratic_objective_term->d_middle_dense = problem->objective_lowrank_middle_dense;
        problem->objective_lowrank_middle_dense = NULL;
        ALLOC_ZERO(state->quadratic_objective_term->Rx_buffer, (size_t)rank * sizeof(double));
    }
}

static void initialize_quadratic_obj_term(pdhg_solver_state_t *state, processed_qp_problem_t *problem)
{
    state->quadratic_objective_term = (quadratic_objective_term_t *)safe_calloc(1, sizeof(quadratic_objective_term_t));
    state->quadratic_objective_term->quad_obj_type = problem->quad_type;
    state->quadratic_objective_term->num_rank_lowrank_obj = problem->num_rank_lowrank_obj;

    if (state->quadratic_objective_term->quad_obj_type == PDHCG_NON_Q)
        return;

    int n_local = problem->num_variables;
    int n_global = get_global_n(state);
    int n_start = get_n_start(state->grid_context);

    size_t alloc_elements = (n_global > n_local) ? n_global : n_local;
    ALLOC_ZERO(state->quadratic_objective_term->global_primal_obj_product, alloc_elements * sizeof(double));

    state->quadratic_objective_term->primal_obj_product =
        state->quadratic_objective_term->global_primal_obj_product + n_start;

    DEVICE_CHECK(pdhcg_device_vector_create(&state->quadratic_objective_term->vec_primal_obj_prod,
                                            n_local,
                                            state->quadratic_objective_term->primal_obj_product));

    if (n_global > n_local)
    {
        DEVICE_CHECK(pdhcg_device_vector_create(&state->quadratic_objective_term->vec_global_primal_obj_prod,
                                                n_global,
                                                state->quadratic_objective_term->global_primal_obj_product));
    }
    else
    {
        state->quadratic_objective_term->vec_global_primal_obj_prod =
            state->quadratic_objective_term->vec_primal_obj_prod;
    }

    switch (state->quadratic_objective_term->quad_obj_type)
    {
        case PDHCG_DIAG_Q:
        {
            state->quadratic_objective_term->diagonal_objective_matrix = problem->diagonal_quad_objective;
            problem->diagonal_quad_objective = NULL;
            state->quadratic_objective_term->objective_sparse_matrix = NULL;
            state->quadratic_objective_term->objective_lowrank_matrix = NULL;
            break;
        }

        case PDHCG_SPARSE_Q:
        {
            initialize_sparse_component_obj(state, problem);
            state->quadratic_objective_term->diagonal_objective_matrix = NULL;
            break;
        }

        case PDHCG_LOW_RANK_Q:
        {
            initialize_lowrank_component_obj(state, problem);
            state->quadratic_objective_term->diagonal_objective_matrix = NULL;
            break;
        }

        case PDHCG_LOW_RANK_PLUS_SPARSE_Q:
        {
            initialize_sparse_component_obj(state, problem);
            initialize_lowrank_component_obj(state, problem);
            state->quadratic_objective_term->diagonal_objective_matrix = NULL;
            break;
        }

        default:
            fprintf(stderr, "Error: Unknown Quadratic Objective Type detected.\n");
            exit(EXIT_FAILURE);
    }
}

static void initialize_inner_solver(pdhg_solver_state_t *state, const pdhg_parameters_t *params)
{
    state->inner_solver = (inner_solver_t *)safe_calloc(1, sizeof(inner_solver_t));

    int iteration_limit = params->inner_solver_parameters.iteration_limit;
    double initial_tol = params->inner_solver_parameters.initial_tolerance;
    double min_tol = params->inner_solver_parameters.min_tolerance;

    if (iteration_limit < 1)
    {
        fprintf(stderr, "Warning: inner_iter_limit (%d) < 1. Resetting to 1.\n", iteration_limit);
        iteration_limit = 1;
    }

    if (initial_tol <= 0.0)
    {
        fprintf(stderr, "Warning: inner_init_tol (%.1e) <= 0. Resetting to default 1e-3.\n", initial_tol);
        initial_tol = 1e-3;
    }

    if (min_tol <= 0.0)
    {
        fprintf(stderr, "Warning: inner_min_tol (%.1e) <= 0. Resetting to default 1e-9.\n", min_tol);
        min_tol = 1e-9;
    }

    state->inner_solver->iteration_limit = iteration_limit;
    state->inner_solver->tol = initial_tol;
    state->inner_solver->min_tol = min_tol;

    if (state->quadratic_objective_term->quad_obj_type == PDHCG_NON_Q || state->use_linearized_quadratic_update ||
        (state->quadratic_objective_term->quad_obj_type == PDHCG_DIAG_Q && !state->cones.has_psd_cones))
        return;

    /* A nonuniform diagonal metric has no one-EVD PSD prox, so diagonal-Q
       models with PSD variables use the existing projected BB solve. */
    ALLOC_ZERO(state->inner_solver->primal_buffer, state->num_variables * sizeof(double));
    ALLOC_ZERO(state->inner_solver->dual_buffer, state->num_constraints * sizeof(double));
    state->inner_solver->bb_step_size = (bb_step_size_t *)safe_calloc(1, sizeof(bb_step_size_t));
    ALLOC_ZERO(state->inner_solver->bb_step_size->gradient, state->num_variables * sizeof(double));
    ALLOC_ZERO(state->inner_solver->bb_step_size->direction, state->num_variables * sizeof(double));
    ALLOC_ZERO(state->inner_solver->bb_step_size->scalar_buffer, 4 * sizeof(double));

    state->inner_solver->bb_step_size->precond_enabled = params->diag_jacobi_precond && !state->cones.has_psd_cones;
    if (state->inner_solver->bb_step_size->precond_enabled)
    {
        int n = state->num_variables;
        ALLOC_ZERO(state->inner_solver->bb_step_size->diag_h_static, n * sizeof(double));
        ALLOC_ZERO(state->inner_solver->bb_step_size->m_diag, n * sizeof(double));
        ALLOC_ZERO(state->inner_solver->bb_step_size->m_inv, n * sizeof(double));
        ALLOC_ZERO(state->inner_solver->bb_step_size->Ms_buffer, n * sizeof(double));
        state->inner_solver->bb_step_size->cached_inv_tau = -1.0;
        state->inner_solver->bb_step_size->tol_scale = 1.0;

        if (quadratic_type_has_sparse_component(state->quadratic_objective_term->quad_obj_type))
        {
            device_sparse_matrix_csr_t *Q = state->quadratic_objective_term->objective_sparse_matrix;
            pdhcg_device_compute_csr_diag(
                Q->row_ptr, Q->col_ind, Q->val, state->inner_solver->bb_step_size->diag_h_static, n);
            DEVICE_CHECK(pdhcg_device_last_error());
        }

        if (quadratic_type_has_lowrank_component(state->quadratic_objective_term->quad_obj_type))
        {
            device_sparse_matrix_csr_t *Rt = state->quadratic_objective_term->objective_lowrank_matrix_t;
            double *out = state->inner_solver->bb_step_size->Ms_buffer;
            int mtype = state->quadratic_objective_term->lowrank_middle_type;
            if (mtype == 1)
            {
                pdhcg_device_compute_csr_row_sq_norm_weighted(
                    Rt->row_ptr, Rt->col_ind, Rt->val, state->quadratic_objective_term->d_middle_diag, out, n);
            }
            else if (mtype == 2)
            {
                int rank = state->quadratic_objective_term->num_rank_lowrank_obj;
                pdhcg_device_compute_csr_row_quad_form_dense(
                    Rt->row_ptr, Rt->col_ind, Rt->val, state->quadratic_objective_term->d_middle_dense, rank, out, n);
            }
            else
            {
                pdhcg_device_compute_csr_row_sq_norm(Rt->row_ptr, Rt->val, out, n);
            }
            DEVICE_CHECK(pdhcg_device_last_error());
            const double one = 1.0;
            DEVICE_CHECK(pdhcg_device_axpy(
                state->blas_handle, n, &one, out, 1, state->inner_solver->bb_step_size->diag_h_static, 1));
            DEVICE_CHECK(pdhcg_device_zero(out, 0, n * sizeof(double)));
        }
    }
}

static void decide_problem_type(pdhg_solver_state_t *state)
{
    if (state->quadratic_objective_term->quad_obj_type == PDHCG_NON_Q)
        state->problem_type = LP;
    else
        state->problem_type = CONVEX_QP;
}

void initialize_quadratic_term_information(pdhg_solver_state_t *state, const pdhg_parameters_t *params)
{
    quadratic_objective_term_t *quadratic_objective = state->quadratic_objective_term;
    if (quadratic_objective->quad_obj_type == PDHCG_NON_Q)
        return;

    if (quadratic_objective->quad_obj_type == PDHCG_DIAG_Q)
    {
        double max_eigen = 0.0;
        double min_eigen = INFINITY;
        double *temp_diag_host = (double *)safe_malloc((size_t)state->num_variables * sizeof(double));
        DEVICE_CHECK(pdhcg_device_copy(temp_diag_host,
                                       quadratic_objective->diagonal_objective_matrix,
                                       (size_t)state->num_variables * sizeof(double),
                                       PDHCG_COPY_DEVICE_TO_HOST));
        for (int i = 0; i < state->num_variables; i++)
        {
            double item = temp_diag_host[i];
            max_eigen = fmax(max_eigen, fabs(item));
            min_eigen = fmin(min_eigen, item);
        }
        quadratic_objective->norm = max_eigen;
        quadratic_objective->nonconvexity = state->num_variables > 0 ? min_eigen : 0.0;
        free(temp_diag_host);
        return;
    }

    quadratic_objective->norm = estimate_quadratic_objective_norm(state->sparse_handle,
                                                                  state->blas_handle,
                                                                  quadratic_objective,
                                                                  state->num_variables,
                                                                  params->sv_max_iter,
                                                                  params->sv_tol,
                                                                  state->grid_context);
    quadratic_objective->nonconvexity = estimate_quadratic_objective_minimum_eigenvalue(state->sparse_handle,
                                                                                        state->blas_handle,
                                                                                        quadratic_objective,
                                                                                        state->num_variables,
                                                                                        quadratic_objective->norm,
                                                                                        params->sv_max_iter,
                                                                                        params->sv_tol,
                                                                                        state->grid_context);
}

static void
initialize_cone_layout(cone_runtime_t *runtime, const cone_blocks_t *cones, const double *coordinate_rescaling)
{
    runtime->num_blocks = cones->num_cones;
    if (runtime->num_blocks == 0)
        return;

    int K = runtime->num_blocks;

    cone_proj_method_t *methods = (cone_proj_method_t *)safe_malloc(K * sizeof(cone_proj_method_t));
    pdhcg_device_cone_methods(cones, coordinate_rescaling, methods);

    int bucket_count[NUM_CONE_TYPES][NUM_PROJ_METHODS] = {{0}};
    for (int i = 0; i < K; ++i)
        bucket_count[cones->type[i]][methods[i]]++;
    for (int method = 0; method < NUM_PROJ_METHODS; ++method)
        runtime->has_power_cones |= bucket_count[CONE_POWER][method] > 0;
    runtime->has_psd_cones = bucket_count[CONE_PSD][PROJ_METHOD_THREAD] > 0;

    cone_bucket_t buckets_tmp[NUM_CONE_TYPES * NUM_PROJ_METHODS];
    int num_buckets = 0;
    int offset = 0;
    int bucket_offset[NUM_CONE_TYPES][NUM_PROJ_METHODS];
    for (int t = 0; t < NUM_CONE_TYPES; ++t)
    {
        for (int m = 0; m < NUM_PROJ_METHODS; ++m)
        {
            bucket_offset[t][m] = offset;
            if (bucket_count[t][m] > 0 && t != CONE_PSD)
            {
                buckets_tmp[num_buckets].type = (cone_type_t)t;
                buckets_tmp[num_buckets].method = (cone_proj_method_t)m;
                buckets_tmp[num_buckets].offset = offset;
                buckets_tmp[num_buckets].count = bucket_count[t][m];
                num_buckets++;
            }
            offset += bucket_count[t][m];
        }
    }

    runtime->num_buckets = num_buckets;
    if (num_buckets > 0)
    {
        runtime->buckets = (cone_bucket_t *)safe_malloc((size_t)num_buckets * sizeof(cone_bucket_t));
        memcpy(runtime->buckets, buckets_tmp, (size_t)num_buckets * sizeof(cone_bucket_t));
    }

    size_t cb = (size_t)K * sizeof(int);
    int *start_perm = (int *)safe_malloc(cb);
    int *vdim_perm = (int *)safe_malloc(cb);
    double *alpha_perm = NULL;
    if (cones->power_alpha)
        alpha_perm = (double *)safe_malloc((size_t)K * sizeof(double));
    int write_pos[NUM_CONE_TYPES][NUM_PROJ_METHODS];
    memcpy(write_pos, bucket_offset, sizeof(write_pos));
    for (int i = 0; i < K; ++i)
    {
        int t = cones->type[i];
        int m = methods[i];
        int p = write_pos[t][m]++;
        start_perm[p] = cones->start_idx[i];
        vdim_perm[p] = cones->v_dim[i];
        if (alpha_perm)
            alpha_perm[p] = cones->power_alpha[i];
    }

    DEVICE_CHECK(pdhcg_device_allocate((void **)&runtime->start_idx, cb));
    DEVICE_CHECK(pdhcg_device_copy(runtime->start_idx, start_perm, cb, PDHCG_COPY_HOST_TO_DEVICE));
    DEVICE_CHECK(pdhcg_device_allocate((void **)&runtime->v_dim, cb));
    DEVICE_CHECK(pdhcg_device_copy(runtime->v_dim, vdim_perm, cb, PDHCG_COPY_HOST_TO_DEVICE));
    if (alpha_perm)
    {
        size_t ab = (size_t)K * sizeof(double);
        DEVICE_CHECK(pdhcg_device_allocate((void **)&runtime->power_alpha, ab));
        DEVICE_CHECK(pdhcg_device_copy(runtime->power_alpha, alpha_perm, ab, PDHCG_COPY_HOST_TO_DEVICE));
        free(alpha_perm);
    }
    if (runtime->has_psd_cones)
    {
        int psd_offset = bucket_offset[CONE_PSD][PROJ_METHOD_THREAD];
        int psd_count = bucket_count[CONE_PSD][PROJ_METHOD_THREAD];
        runtime->psd =
            create_psd_projection_runtime(start_perm + psd_offset, vdim_perm + psd_offset, psd_count, psd_offset);
    }
    free(start_perm);
    free(vdim_perm);
    free(methods);

    size_t scalar_bytes = (size_t)K * sizeof(double);
    size_t workspace_bytes = PDHCG_CONE_WORKSPACE_STRIDE * scalar_bytes;
    DEVICE_CHECK(pdhcg_device_allocate((void **)&runtime->projection_workspace, workspace_bytes));
    DEVICE_CHECK(pdhcg_device_zero(runtime->projection_workspace, 0, workspace_bytes));
    DEVICE_CHECK(pdhcg_device_allocate((void **)&runtime->residual_workspace, workspace_bytes));
    DEVICE_CHECK(pdhcg_device_zero(runtime->residual_workspace, 0, workspace_bytes));
    DEVICE_CHECK(pdhcg_device_allocate((void **)&runtime->infeasibility_workspace, workspace_bytes));
    DEVICE_CHECK(pdhcg_device_zero(runtime->infeasibility_workspace, 0, workspace_bytes));
    DEVICE_CHECK(pdhcg_device_allocate((void **)&runtime->complementarity_residual, scalar_bytes));
    DEVICE_CHECK(pdhcg_device_zero(runtime->complementarity_residual, 0, scalar_bytes));
    if (runtime->axis == CONE_AXIS_VARIABLE && runtime->has_power_cones)
        DEVICE_CHECK(pdhcg_device_allocate((void **)&runtime->power_violation_workspace, 2 * scalar_bytes));
}

static void initialize_infeasibility_cone_type(pdhg_solver_state_t *state, const cone_blocks_t *cones)
{
    if (!state->has_variable_cones || state->num_variables <= 0)
        return;

    size_t bytes = (size_t)state->num_variables * sizeof(unsigned char);
    unsigned char *type = (unsigned char *)safe_calloc((size_t)state->num_variables, sizeof(unsigned char));
    for (int cone = 0; cone < cones->num_cones; ++cone)
    {
        int start = cones->start_idx[cone];
        int length = cone_block_length(cones, cone);
        unsigned char cone_type = 1;
        if (cones->is_fixed)
        {
            for (int slot = start; slot < start + length; ++slot)
                if (cones->is_fixed[slot])
                    cone_type = 2;
        }
        memset(type + start, cone_type, (size_t)length * sizeof(unsigned char));
    }
    /* A fixed slot on another rank still makes the whole split cone a section. */
    initialize_split_cone_infeasibility_type(state, type);
    ALLOC_AND_COPY(state->cones.infeasibility_type, type, bytes);
    free(type);
}

static void initialize_cone_runtime(pdhg_solver_state_t *state, const qp_problem_t *working_problem)
{
    memset(&state->cones, 0, sizeof(state->cones));
    memset(&state->affine_cones, 0, sizeof(state->affine_cones));
    state->cones.axis = CONE_AXIS_VARIABLE;
    state->affine_cones.axis = CONE_AXIS_CONSTRAINT;

    initialize_split_cones(state);

    bool has_global_cones = working_problem->cones.num_cones > 0 || state->cones.split != NULL ||
        pdhcg_get_global_num_cones(state->grid_context) > 0;
    state->has_variable_cones = has_global_cones;
    initialize_infeasibility_cone_type(state, &working_problem->cones);

    if (working_problem->cones.is_fixed)
    {
        size_t fb = (size_t)state->num_variables * sizeof(char);
        DEVICE_CHECK(pdhcg_device_allocate((void **)&state->cones.is_fixed, fb));
        DEVICE_CHECK(
            pdhcg_device_copy(state->cones.is_fixed, working_problem->cones.is_fixed, fb, PDHCG_COPY_HOST_TO_DEVICE));
    }

    initialize_cone_layout(&state->cones, &working_problem->cones, state->variable_rescaling);
    double global_has_power_cones = state->cones.has_power_cones ? 1.0 : 0.0;
    pdhcg_all_reduce_scalar(state->grid_context, &global_has_power_cones, PDHCG_OP_MAX, PDHCG_SCOPE_ROW, false);
    state->cones.has_power_cones = global_has_power_cones != 0.0;
    double global_has_psd_cones = state->cones.has_psd_cones ? 1.0 : 0.0;
    pdhcg_all_reduce_scalar(state->grid_context, &global_has_psd_cones, PDHCG_OP_MAX, PDHCG_SCOPE_ROW, false);
    state->cones.has_psd_cones = global_has_psd_cones != 0.0;
    if (state->has_variable_cones)
    {
        quad_obj_type_t qt = state->quadratic_objective_term->quad_obj_type;
        size_t variable_bytes = (size_t)state->num_variables * sizeof(double);
        if (qt != PDHCG_NON_Q)
            DEVICE_CHECK(pdhcg_device_allocate((void **)&state->cones.effective_objective_gradient, variable_bytes));
        if (qt != PDHCG_NON_Q && (qt != PDHCG_DIAG_Q || state->cones.has_psd_cones))
            DEVICE_CHECK(pdhcg_device_allocate((void **)&state->cones.bb_primal_snapshot, variable_bytes));
    }

    bool has_affine_cones = working_problem->affine_cones.num_cones > 0 || state->affine_cones.split != NULL ||
        pdhcg_get_global_num_affine_cones(state->grid_context) > 0;

    if (has_affine_cones && state->num_constraints > 0)
    {
        size_t constraint_bytes = (size_t)state->num_constraints * sizeof(double);
        DEVICE_CHECK(pdhcg_device_allocate((void **)&state->affine_cones.coordinate_rescaling, constraint_bytes));
    }
    pdhcg_device_initialize_cone_scaling_and_primal(state);
    initialize_cone_layout(
        &state->affine_cones, &working_problem->affine_cones, state->affine_cones.coordinate_rescaling);
}

pdhg_solver_state_t *initialize_solver_state(const pdhg_parameters_t *params,
                                             const qp_problem_t *working_problem,
                                             const qp_problem_t *device_problem,
                                             grid_context_t *grid_context)
{
    pdhg_solver_state_t *state = (pdhg_solver_state_t *)safe_calloc(1, sizeof(pdhg_solver_state_t));

    state->grid_context = grid_context;

    int n_vars = device_problem->num_variables;
    int n_cons = device_problem->num_constraints;
    size_t var_bytes = n_vars * sizeof(double);
    size_t con_bytes = n_cons * sizeof(double);

    state->num_variables = n_vars;
    state->num_constraints = n_cons;
    state->objective_constant = device_problem->objective_constant;

    DEVICE_CHECK(pdhcg_device_sparse_create(&state->sparse_handle));
    DEVICE_CHECK(pdhcg_device_blas_create(&state->blas_handle));
    DEVICE_CHECK(pdhcg_device_set_pointer_mode(state->blas_handle, PDHCG_POINTER_HOST));
    DEVICE_CHECK(pdhcg_device_allocate((void **)&state->constraint_rescaling, con_bytes));
    DEVICE_CHECK(pdhcg_device_allocate((void **)&state->variable_rescaling, var_bytes));

    quad_obj_type_t quad_type;
#ifdef PDHCG_COMPILE_DISTRIBUTED
    if (grid_context)
        quad_type = grid_context->quad_type;
    else
#endif
        quad_type = detect_q_type(working_problem->objective_sparse_matrix,
                                  working_problem->objective_lowrank_matrix,
                                  n_vars,
                                  working_problem->num_rank_lowrank_obj);

    processed_qp_problem_t *processed =
        preprocess_qp_problem(working_problem, device_problem, quad_type, get_n_start(grid_context));

    state->constraint_matrix = (device_sparse_matrix_csr_t *)safe_malloc(sizeof(device_sparse_matrix_csr_t));

    state->constraint_matrix->num_rows = n_cons;
    state->constraint_matrix->num_cols = n_vars;
    state->constraint_matrix->num_nonzeros = device_problem->constraint_matrix_num_nonzeros;

    state->termination_reason = TERMINATION_REASON_UNSPECIFIED;

    ALLOC_AND_COPY_DEVICE_CSR(state->constraint_matrix,
                              device_problem->constraint_matrix,
                              device_problem->num_constraints,
                              device_problem->constraint_matrix_num_nonzeros);

    state->constraint_matrix_t = (device_sparse_matrix_csr_t *)safe_malloc(sizeof(device_sparse_matrix_csr_t));
    state->constraint_matrix_t->num_rows = n_vars;
    state->constraint_matrix_t->num_cols = n_cons;
    state->constraint_matrix_t->num_nonzeros = state->constraint_matrix->num_nonzeros;
    DEVICE_CHECK(
        pdhcg_device_allocate((void **)&state->constraint_matrix_t->row_ptr, ((size_t)n_vars + 1) * sizeof(int)));
    DEVICE_CHECK(pdhcg_device_allocate((void **)&state->constraint_matrix_t->col_ind,
                                       (size_t)state->constraint_matrix->num_nonzeros * sizeof(int)));
    DEVICE_CHECK(pdhcg_device_allocate((void **)&state->constraint_matrix_t->val,
                                       (size_t)state->constraint_matrix->num_nonzeros * sizeof(double)));
    DEVICE_CHECK(pdhcg_device_csr_transpose(state->sparse_handle,
                                            state->constraint_matrix->num_rows,
                                            state->constraint_matrix->num_cols,
                                            state->constraint_matrix->num_nonzeros,
                                            state->constraint_matrix->val,
                                            state->constraint_matrix->row_ptr,
                                            state->constraint_matrix->col_ind,
                                            state->constraint_matrix_t->val,
                                            state->constraint_matrix_t->row_ptr,
                                            state->constraint_matrix_t->col_ind));
    ALLOC_AND_COPY_DEVICE(state->variable_lower_bound, device_problem->variable_lower_bound, var_bytes);
    ALLOC_AND_COPY_DEVICE(state->variable_upper_bound, device_problem->variable_upper_bound, var_bytes);
    ALLOC_AND_COPY_DEVICE(state->objective_vector, device_problem->objective_vector, var_bytes);
    ALLOC_AND_COPY_DEVICE(state->constraint_lower_bound, device_problem->constraint_lower_bound, con_bytes);
    ALLOC_AND_COPY_DEVICE(state->constraint_upper_bound, device_problem->constraint_upper_bound, con_bytes);
    ALLOC_AND_COPY_DEVICE(state->affine_cone_offset, device_problem->affine_cone_offset, con_bytes);

    if (device_problem->primal_start)
    {
        ALLOC_AND_COPY_DEVICE(state->initial_primal_solution, device_problem->primal_start, var_bytes);
    }
    else
    {
        ALLOC_ZERO(state->initial_primal_solution, var_bytes);
    }
    if (device_problem->dual_start)
    {
        ALLOC_AND_COPY_DEVICE(state->initial_dual_solution, device_problem->dual_start, con_bytes);
    }
    else
    {
        ALLOC_ZERO(state->initial_dual_solution, con_bytes);
    }

    initialize_quadratic_obj_term(state, processed);
    free_processed_qp_problem(processed);

    /* Scale state storage before creating SpMV contexts and derived solver data. */
    double start_rescaling = monotonic_time_seconds();
    pdhcg_device_rescale_problem(params, device_problem, state);
    DEVICE_CHECK(pdhcg_device_last_error());
    DEVICE_CHECK(pdhcg_device_synchronize());
    state->rescaling_time_sec = monotonic_time_seconds() - start_rescaling;

    ALLOC_ZERO(state->current_primal_solution, var_bytes);
    ALLOC_ZERO(state->pdhg_primal_solution, var_bytes);
    ALLOC_ZERO(state->reflected_primal_solution, var_bytes);
    ALLOC_ZERO(state->dual_product, var_bytes);
    ALLOC_ZERO(state->dual_slack, var_bytes);
    ALLOC_ZERO(state->infeasibility_dual_workspace, var_bytes);
    ALLOC_ZERO(state->dual_residual, var_bytes);
    ALLOC_ZERO(state->delta_primal_solution, var_bytes);

    ALLOC_ZERO(state->current_dual_solution, con_bytes);
    ALLOC_ZERO(state->pdhg_dual_solution, con_bytes);
    ALLOC_ZERO(state->reflected_dual_solution, con_bytes);
    ALLOC_ZERO(state->primal_product, con_bytes);
    ALLOC_ZERO(state->primal_slack, con_bytes);
    ALLOC_ZERO(state->primal_residual, con_bytes);
    ALLOC_ZERO(state->delta_dual_solution, con_bytes);

    DEVICE_CHECK(pdhcg_device_copy(
        state->current_dual_solution, state->initial_dual_solution, con_bytes, PDHCG_COPY_DEVICE_TO_DEVICE));
    DEVICE_CHECK(pdhcg_device_copy(
        state->pdhg_dual_solution, state->initial_dual_solution, con_bytes, PDHCG_COPY_DEVICE_TO_DEVICE));
    pdhcg_device_initialize_finite_bounds(state);

    double sum_of_squares = 0.0;
    double max_val = 0.0;
    double val = 0.0;

    for (int i = 0; i < n_vars; ++i)
    {
        if (params->optimality_norm == NORM_TYPE_L_INF)
        {
            val = fabs(working_problem->objective_vector[i]);
            if (val > max_val)
                max_val = val;
        }
        else
        {
            sum_of_squares += working_problem->objective_vector[i] * working_problem->objective_vector[i];
        }
    }

    if (params->optimality_norm == NORM_TYPE_L_INF)
    {
        state->objective_vector_norm = max_val;
    }
    else
    {
        state->objective_vector_norm = sqrt(sum_of_squares);
    }

    sum_of_squares = 0.0;
    max_val = 0.0;
    val = 0.0;

    for (int i = 0; i < n_cons; ++i)
    {
        double lower = working_problem->constraint_lower_bound[i];
        double upper = working_problem->constraint_upper_bound[i];

        if (params->optimality_norm == NORM_TYPE_L_INF)
        {
            if (isfinite(lower) && (lower != upper))
            {
                val = fabs(lower);
                if (val > max_val)
                    max_val = val;
            }
            if (isfinite(upper))
            {
                val = fabs(upper);
                if (val > max_val)
                    max_val = val;
            }
        }
        else
        {
            if (isfinite(lower) && (lower != upper))
            {
                sum_of_squares += lower * lower;
            }
            if (isfinite(upper))
            {
                sum_of_squares += upper * upper;
            }
        }
    }

    for (int i = 0; i < working_problem->num_constraints; ++i)
    {
        double constant = working_problem->affine_cone_offset[i];
        if (params->optimality_norm == NORM_TYPE_L_INF)
            max_val = fmax(max_val, fabs(constant));
        else
            sum_of_squares += constant * constant;
    }

    if (params->optimality_norm == NORM_TYPE_L_INF)
    {
        state->constraint_bound_norm = max_val;
    }
    else
    {
        state->constraint_bound_norm = sqrt(sum_of_squares);
    }

    state->best_primal_dual_residual_gap = INFINITY;
    state->last_trial_fixed_point_error = INFINITY;
    state->step_size = 0.0;
    state->is_this_major_iteration = false;

    DEVICE_CHECK(pdhcg_device_vector_create(&state->vec_primal_sol, state->num_variables, state->pdhg_primal_solution));
    DEVICE_CHECK(pdhcg_device_vector_create(&state->vec_dual_sol, state->num_constraints, state->pdhg_dual_solution));
    DEVICE_CHECK(pdhcg_device_vector_create(&state->vec_primal_prod, state->num_constraints, state->primal_product));
    DEVICE_CHECK(pdhcg_device_vector_create(&state->vec_dual_prod, state->num_variables, state->dual_product));

    state->spmv_ctx_A = pdhcg_spmv_ctx_create(state->sparse_handle,
                                              state->num_constraints,
                                              state->num_variables,
                                              state->constraint_matrix->num_nonzeros,
                                              state->constraint_matrix->row_ptr,
                                              state->constraint_matrix->col_ind,
                                              state->constraint_matrix->val,
                                              state->vec_primal_sol,
                                              state->vec_primal_prod);

    state->spmv_ctx_At = pdhcg_spmv_ctx_create(state->sparse_handle,
                                               state->num_variables,
                                               state->num_constraints,
                                               state->constraint_matrix_t->num_nonzeros,
                                               state->constraint_matrix_t->row_ptr,
                                               state->constraint_matrix_t->col_ind,
                                               state->constraint_matrix_t->val,
                                               state->vec_dual_sol,
                                               state->vec_dual_prod);

    quadratic_objective_term_t *term = state->quadratic_objective_term;
    if (quadratic_type_has_sparse_component(term->quad_obj_type))
    {
        term->spmv_ctx_Q = pdhcg_spmv_ctx_create(state->sparse_handle,
                                                 term->objective_sparse_matrix->num_rows,
                                                 state->num_variables,
                                                 term->objective_sparse_matrix->num_nonzeros,
                                                 term->objective_sparse_matrix->row_ptr,
                                                 term->objective_sparse_matrix->col_ind,
                                                 term->objective_sparse_matrix->val,
                                                 state->vec_primal_sol,
                                                 term->vec_global_primal_obj_prod);
    }
    if (quadratic_type_has_lowrank_component(term->quad_obj_type))
    {
        term->spmv_ctx_R = pdhcg_spmv_ctx_create(state->sparse_handle,
                                                 term->num_rank_lowrank_obj,
                                                 state->num_variables,
                                                 term->objective_lowrank_matrix->num_nonzeros,
                                                 term->objective_lowrank_matrix->row_ptr,
                                                 term->objective_lowrank_matrix->col_ind,
                                                 term->objective_lowrank_matrix->val,
                                                 state->vec_primal_sol,
                                                 term->vec_Rx_prod);

        term->spmv_ctx_Rt = pdhcg_spmv_ctx_create(state->sparse_handle,
                                                  state->num_variables,
                                                  term->num_rank_lowrank_obj,
                                                  term->objective_lowrank_matrix_t->num_nonzeros,
                                                  term->objective_lowrank_matrix_t->row_ptr,
                                                  term->objective_lowrank_matrix_t->col_ind,
                                                  term->objective_lowrank_matrix_t->val,
                                                  term->vec_Rx_prod,
                                                  term->vec_primal_obj_prod);
    }

    state->num_original_variables = working_problem->num_original_variables;
    initialize_cone_runtime(state, working_problem);
    if (state->has_variable_cones)
    {
        project_cone_runtime(state, &state->cones, state->initial_primal_solution, state->cones.projection_workspace);
    }
    if (state->num_variables > 0)
    {
        pdhcg_device_project_primal_onto_bounds(state->initial_primal_solution,
                                                state->variable_lower_bound,
                                                state->variable_upper_bound,
                                                state->num_variables);
    }
    DEVICE_CHECK(pdhcg_device_last_error());
    DEVICE_CHECK(pdhcg_device_copy(
        state->current_primal_solution, state->initial_primal_solution, var_bytes, PDHCG_COPY_DEVICE_TO_DEVICE));
    DEVICE_CHECK(pdhcg_device_copy(
        state->pdhg_primal_solution, state->initial_primal_solution, var_bytes, PDHCG_COPY_DEVICE_TO_DEVICE));
    DEVICE_CHECK(pdhcg_device_copy(
        state->reflected_primal_solution, state->initial_primal_solution, var_bytes, PDHCG_COPY_DEVICE_TO_DEVICE));

    state->use_linearized_quadratic_update = uses_linearized_quadratic_update(
        params->non_diagonal_quadratic_mode, state->quadratic_objective_term->quad_obj_type);
    initialize_quadratic_term_information(state, params);
    initialize_inner_solver(state, params);

    DEVICE_CHECK(pdhcg_device_allocate((void **)&state->ones_primal, state->num_variables * sizeof(double)));
    DEVICE_CHECK(pdhcg_device_allocate((void **)&state->ones_dual, state->num_constraints * sizeof(double)));

    double *ones_primal_h = (double *)safe_malloc(state->num_variables * sizeof(double));
    for (int i = 0; i < state->num_variables; ++i)
        ones_primal_h[i] = 1.0;
    DEVICE_CHECK(pdhcg_device_copy(
        state->ones_primal, ones_primal_h, state->num_variables * sizeof(double), PDHCG_COPY_HOST_TO_DEVICE));
    free(ones_primal_h);

    double *ones_dual_h = (double *)safe_malloc(state->num_constraints * sizeof(double));
    for (int i = 0; i < state->num_constraints; ++i)
        ones_dual_h[i] = 1.0;
    DEVICE_CHECK(pdhcg_device_copy(
        state->ones_dual, ones_dual_h, state->num_constraints * sizeof(double), PDHCG_COPY_HOST_TO_DEVICE));
    decide_problem_type(state);
    free(ones_dual_h);
    if (params->verbose >= 2)
    {
        printf("-------------------------------------------------------------------"
               "----------"
               "----------------------\n");
        printf("Problem Type: %s\n", problem_type_to_string(state->problem_type));
        printf("Quadratic Objective Matrix Type: %s\n",
               quad_obj_type_to_string(state->quadratic_objective_term->quad_obj_type));
        if (state->quadratic_objective_term->quad_obj_type != PDHCG_NON_Q)
        {
            printf("Quadratic Objective Norm Estimate: %.3e\n", state->quadratic_objective_term->norm);
            printf("Quadratic Objective Minimum Eigenvalue Estimate: %.3e\n",
                   state->quadratic_objective_term->nonconvexity);
        }
        printf("-------------------------------------------------------------------"
               "----------"
               "----------------------\n");
        printf("%s | %s | %s | %s \n",
               "        runtime       ",
               "    objective     ",
               "  absolute residuals   ",
               "  relative residuals   ");
        printf("%s %s %s | %s %s | %s %s %s | %s %s %s \n",
               "  iter",
               "  inner",
               "  time ",
               " pr obj ",
               "  du obj ",
               " pr res",
               " du res",
               "  gap  ",
               " pr res",
               " du res",
               "  gap  ");
        printf("-------------------------------------------------------------------"
               "----------"
               "----------------------\n");
    }
    return state;
}

void pdhg_solver_state_free(pdhg_solver_state_t *state)
{
    if (state == NULL)
    {
        return;
    }

    if (state->variable_lower_bound)
        DEVICE_CHECK(pdhcg_device_free(state->variable_lower_bound));
    if (state->variable_upper_bound)
        DEVICE_CHECK(pdhcg_device_free(state->variable_upper_bound));
    if (state->objective_vector)
        DEVICE_CHECK(pdhcg_device_free(state->objective_vector));
    if (state->constraint_lower_bound)
        DEVICE_CHECK(pdhcg_device_free(state->constraint_lower_bound));
    if (state->constraint_upper_bound)
        DEVICE_CHECK(pdhcg_device_free(state->constraint_upper_bound));
    if (state->affine_cone_offset)
        DEVICE_CHECK(pdhcg_device_free(state->affine_cone_offset));
    if (state->constraint_lower_bound_finite_val)
        DEVICE_CHECK(pdhcg_device_free(state->constraint_lower_bound_finite_val));
    if (state->constraint_upper_bound_finite_val)
        DEVICE_CHECK(pdhcg_device_free(state->constraint_upper_bound_finite_val));
    if (state->variable_lower_bound_finite_val)
        DEVICE_CHECK(pdhcg_device_free(state->variable_lower_bound_finite_val));
    if (state->variable_upper_bound_finite_val)
        DEVICE_CHECK(pdhcg_device_free(state->variable_upper_bound_finite_val));
    if (state->initial_primal_solution)
        DEVICE_CHECK(pdhcg_device_free(state->initial_primal_solution));
    if (state->current_primal_solution)
        DEVICE_CHECK(pdhcg_device_free(state->current_primal_solution));
    if (state->pdhg_primal_solution)
        DEVICE_CHECK(pdhcg_device_free(state->pdhg_primal_solution));
    if (state->reflected_primal_solution)
        DEVICE_CHECK(pdhcg_device_free(state->reflected_primal_solution));
    if (state->dual_product)
        DEVICE_CHECK(pdhcg_device_free(state->dual_product));
    if (state->initial_dual_solution)
        DEVICE_CHECK(pdhcg_device_free(state->initial_dual_solution));
    if (state->current_dual_solution)
        DEVICE_CHECK(pdhcg_device_free(state->current_dual_solution));
    if (state->pdhg_dual_solution)
        DEVICE_CHECK(pdhcg_device_free(state->pdhg_dual_solution));
    if (state->reflected_dual_solution)
        DEVICE_CHECK(pdhcg_device_free(state->reflected_dual_solution));
    if (state->primal_product)
        DEVICE_CHECK(pdhcg_device_free(state->primal_product));
    if (state->constraint_rescaling)
        DEVICE_CHECK(pdhcg_device_free(state->constraint_rescaling));
    if (state->variable_rescaling)
        DEVICE_CHECK(pdhcg_device_free(state->variable_rescaling));
    if (state->primal_slack)
        DEVICE_CHECK(pdhcg_device_free(state->primal_slack));
    if (state->dual_slack)
        DEVICE_CHECK(pdhcg_device_free(state->dual_slack));
    if (state->infeasibility_dual_workspace)
        DEVICE_CHECK(pdhcg_device_free(state->infeasibility_dual_workspace));
    if (state->primal_residual)
        DEVICE_CHECK(pdhcg_device_free(state->primal_residual));
    if (state->dual_residual)
        DEVICE_CHECK(pdhcg_device_free(state->dual_residual));
    if (state->delta_primal_solution)
        DEVICE_CHECK(pdhcg_device_free(state->delta_primal_solution));
    if (state->delta_dual_solution)
        DEVICE_CHECK(pdhcg_device_free(state->delta_dual_solution));
    if (state->ones_primal)
        DEVICE_CHECK(pdhcg_device_free(state->ones_primal));
    if (state->ones_dual)
        DEVICE_CHECK(pdhcg_device_free(state->ones_dual));

    if (state->quadratic_objective_term)
    {
        if (state->quadratic_objective_term->global_primal_obj_product)
            DEVICE_CHECK(pdhcg_device_free(state->quadratic_objective_term->global_primal_obj_product));

        if (state->quadratic_objective_term->vec_Rx_prod)
            pdhcg_device_vector_destroy(state->quadratic_objective_term->vec_Rx_prod);
        if (state->quadratic_objective_term->vec_primal_obj_prod &&
            state->quadratic_objective_term->vec_primal_obj_prod !=
                state->quadratic_objective_term->vec_global_primal_obj_prod)
        {
            pdhcg_device_vector_destroy(state->quadratic_objective_term->vec_primal_obj_prod);
        }
        if (state->quadratic_objective_term->vec_global_primal_obj_prod)
            pdhcg_device_vector_destroy(state->quadratic_objective_term->vec_global_primal_obj_prod);
        if (state->quadratic_objective_term->vec_primal_obj_prod ==
            state->quadratic_objective_term->vec_global_primal_obj_prod)
        {
            state->quadratic_objective_term->vec_primal_obj_prod = NULL;
        }
        state->quadratic_objective_term->vec_global_primal_obj_prod = NULL;
        if (state->quadratic_objective_term->spmv_ctx_Q)
            pdhcg_spmv_ctx_destroy(state->quadratic_objective_term->spmv_ctx_Q);
        if (state->quadratic_objective_term->spmv_ctx_R)
            pdhcg_spmv_ctx_destroy(state->quadratic_objective_term->spmv_ctx_R);
        if (state->quadratic_objective_term->spmv_ctx_Rt)
            pdhcg_spmv_ctx_destroy(state->quadratic_objective_term->spmv_ctx_Rt);

        free_device_csr(state->quadratic_objective_term->objective_sparse_matrix);
        free_device_csr(state->quadratic_objective_term->objective_lowrank_matrix);
        free_device_csr(state->quadratic_objective_term->objective_lowrank_matrix_t);
        if (state->quadratic_objective_term->diagonal_objective_matrix)
            DEVICE_CHECK(pdhcg_device_free(state->quadratic_objective_term->diagonal_objective_matrix));
        if (state->quadratic_objective_term->Rx_product)
            DEVICE_CHECK(pdhcg_device_free(state->quadratic_objective_term->Rx_product));

        if (state->quadratic_objective_term->d_middle_diag)
            DEVICE_CHECK(pdhcg_device_free(state->quadratic_objective_term->d_middle_diag));
        if (state->quadratic_objective_term->d_middle_dense)
            DEVICE_CHECK(pdhcg_device_free(state->quadratic_objective_term->d_middle_dense));
        if (state->quadratic_objective_term->Rx_buffer)
            DEVICE_CHECK(pdhcg_device_free(state->quadratic_objective_term->Rx_buffer));

        free(state->quadratic_objective_term);
    }

    if (state->vec_primal_sol)
        pdhcg_device_vector_destroy(state->vec_primal_sol);
    if (state->vec_dual_sol)
        pdhcg_device_vector_destroy(state->vec_dual_sol);
    if (state->vec_primal_prod)
        pdhcg_device_vector_destroy(state->vec_primal_prod);
    if (state->vec_dual_prod)
        pdhcg_device_vector_destroy(state->vec_dual_prod);

    if (state->spmv_ctx_A)
        pdhcg_spmv_ctx_destroy(state->spmv_ctx_A);
    if (state->spmv_ctx_At)
        pdhcg_spmv_ctx_destroy(state->spmv_ctx_At);

    free_device_csr(state->constraint_matrix);
    free_device_csr(state->constraint_matrix_t);

    if (state->blas_handle)
        pdhcg_device_blas_destroy(state->blas_handle);
    if (state->sparse_handle)
        pdhcg_device_sparse_destroy(state->sparse_handle);

    if (state->inner_solver)
    {
        if (state->inner_solver->bb_step_size)
        {
            bb_step_size_t *bb = state->inner_solver->bb_step_size;
            if (bb->gradient)
                DEVICE_CHECK(pdhcg_device_free(bb->gradient));
            if (bb->direction)
                DEVICE_CHECK(pdhcg_device_free(bb->direction));
            if (bb->scalar_buffer)
                DEVICE_CHECK(pdhcg_device_free(bb->scalar_buffer));
            if (bb->diag_h_static)
                DEVICE_CHECK(pdhcg_device_free(bb->diag_h_static));
            if (bb->m_diag)
                DEVICE_CHECK(pdhcg_device_free(bb->m_diag));
            if (bb->m_inv)
                DEVICE_CHECK(pdhcg_device_free(bb->m_inv));
            if (bb->Ms_buffer)
                DEVICE_CHECK(pdhcg_device_free(bb->Ms_buffer));
            free(bb);
        }
        if (state->inner_solver->primal_buffer)
            DEVICE_CHECK(pdhcg_device_free(state->inner_solver->primal_buffer));
        if (state->inner_solver->dual_buffer)
            DEVICE_CHECK(pdhcg_device_free(state->inner_solver->dual_buffer));
        free(state->inner_solver);
    }

    cone_runtime_t *runtimes[] = {&state->cones, &state->affine_cones};
    for (int runtime_idx = 0; runtime_idx < 2; ++runtime_idx)
    {
        cone_runtime_t *runtime = runtimes[runtime_idx];
        if (runtime->start_idx)
            DEVICE_CHECK(pdhcg_device_free(runtime->start_idx));
        if (runtime->v_dim)
            DEVICE_CHECK(pdhcg_device_free(runtime->v_dim));
        if (runtime->power_alpha)
            DEVICE_CHECK(pdhcg_device_free(runtime->power_alpha));
        if (runtime->is_fixed)
            DEVICE_CHECK(pdhcg_device_free(runtime->is_fixed));
        if (runtime->infeasibility_type)
            DEVICE_CHECK(pdhcg_device_free(runtime->infeasibility_type));
        if (runtime->buckets)
            free(runtime->buckets);
        if (runtime->projection_workspace)
            DEVICE_CHECK(pdhcg_device_free(runtime->projection_workspace));
        if (runtime->residual_workspace)
            DEVICE_CHECK(pdhcg_device_free(runtime->residual_workspace));
        if (runtime->infeasibility_workspace)
            DEVICE_CHECK(pdhcg_device_free(runtime->infeasibility_workspace));
        if (runtime->complementarity_residual)
            DEVICE_CHECK(pdhcg_device_free(runtime->complementarity_residual));
        if (runtime->power_violation_workspace)
            DEVICE_CHECK(pdhcg_device_free(runtime->power_violation_workspace));
        if (runtime->coordinate_rescaling)
            DEVICE_CHECK(pdhcg_device_free(runtime->coordinate_rescaling));
        if (runtime->effective_objective_gradient)
            DEVICE_CHECK(pdhcg_device_free(runtime->effective_objective_gradient));
        if (runtime->bb_primal_snapshot)
            DEVICE_CHECK(pdhcg_device_free(runtime->bb_primal_snapshot));
        free_psd_projection_runtime(runtime->psd);
    }
    free_split_cones(state);

    free(state);
}
