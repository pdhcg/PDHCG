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

#include "device_linearized.h"
#include "pdhg_core_op.h"
#include "solver_state.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { VARIABLES = 129, CONSTRAINTS = 67 };

static int failures;

static void check(int condition, const char *message)
{
    if (!condition)
    {
        fprintf(stderr, "FAIL: %s\n", message);
        ++failures;
    }
}

static void upload(double *device, const double *host, int count)
{
    DEVICE_CHECK(pdhcg_device_copy(device, host, (size_t)count * sizeof(double), PDHCG_COPY_HOST_TO_DEVICE));
}

static void download(double *host, const double *device, int count)
{
    DEVICE_CHECK(pdhcg_device_copy(host, device, (size_t)count * sizeof(double), PDHCG_COPY_DEVICE_TO_HOST));
}

static void check_vector(const char *name, const double *device, const double *expected, int count)
{
    if (count == 0)
        return;
    double *actual = safe_malloc((size_t)count * sizeof(double));
    download(actual, device, count);
    for (int i = 0; i < count; ++i)
    {
        if (!isfinite(actual[i]) || fabs(actual[i] - expected[i]) > 2e-10 * (1.0 + fabs(expected[i])))
        {
            fprintf(stderr, "FAIL: %s[%d] = %.17g, expected %.17g\n", name, i, actual[i], expected[i]);
            ++failures;
            break;
        }
    }
    free(actual);
}

static void parameters_without_scaling(pdhg_parameters_t *params)
{
    set_default_parameters(params);
    params->verbose = 0;
    params->presolve = false;
    params->non_diagonal_quadratic_mode = NON_DIAGONAL_QUADRATIC_LINEARIZED;
    params->l_inf_ruiz_iterations = 0;
    params->curtis_reid_iterations = 0;
    params->has_pock_chambolle_alpha = false;
    params->bound_objective_rescaling = false;
    params->sv_max_iter = 100;
    params->sv_tol = 1e-6;
}

static qp_problem_t *make_box_problem(bool dense)
{
    int q_row[VARIABLES + 1], q_col[5 * VARIABLES];
    double q_val[5 * VARIABLES];
    int a_row[CONSTRAINTS + 1], a_col[VARIABLES + 33 * CONSTRAINTS];
    double a_val[VARIABLES + 33 * CONSTRAINTS];
    double c[VARIABLES], lower[VARIABLES], upper[VARIABLES], x[VARIABLES];
    double row_lower[CONSTRAINTS], row_upper[CONSTRAINTS], y[CONSTRAINTS];
    int q_nnz = 0, a_nnz = 0;

    /* Strict diagonal dominance makes Q positive definite. Its first row is
       long, while the remaining rows exercise mixed short CSR row lengths. */
    for (int i = 0; i < VARIABLES; ++i)
    {
        q_row[i] = q_nnz;
        for (int j = 0; j < VARIABLES; ++j)
        {
            double value = i == j ? 4.0 : 0.0;
            if (abs(i - j) == 1)
                value += 0.125;
            if (i != j && (i == 0 || j == 0))
                value += 0.007;
            if (value != 0.0)
            {
                q_col[q_nnz] = j;
                q_val[q_nnz++] = value;
            }
        }
        c[i] = 0.04 * (i % 11 - 5);
        lower[i] = i % 7 == 0 ? -INFINITY : -0.8;
        upper[i] = i % 5 == 0 ? INFINITY : 1.1;
        x[i] = 0.05 * (i % 13 - 6);
    }
    q_row[VARIABLES] = q_nnz;

    for (int i = 0; i < CONSTRAINTS; ++i)
    {
        a_row[i] = a_nnz;
        for (int j = 0; j < VARIABLES; ++j)
        {
            double value = 0.0;
            if (i == 0)
                value = 0.02 * (j % 9 - 4);
            else if (i != 1)
            {
                if (dense)
                {
                    /* Keep entries small while varying the row lengths
                       exercised by cuSPARSE across the two matrix products. */
                    if ((j + 3 * i) % VARIABLES < 33)
                        value = 0.012 * (1 + (j + i) % 3) * ((j + i) % 2 ? -1.0 : 1.0);
                }
                else
                {
                    if (j == (3 * i) % VARIABLES)
                        value += 0.4;
                    if (j == (7 * i + 9) % VARIABLES)
                        value -= 0.7;
                }
            }
            if (value != 0.0)
            {
                a_col[a_nnz] = j;
                a_val[a_nnz++] = value;
            }
        }
        row_lower[i] = i % 4 < 2 ? -INFINITY : -0.5;
        row_upper[i] = i % 4 == 1 || i % 4 == 3 ? INFINITY : 0.7;
        y[i] = 0.03 * (i % 7 - 3);
    }
    a_row[CONSTRAINTS] = a_nnz;

    matrix_desc_t Q = {.m = VARIABLES, .n = VARIABLES, .fmt = matrix_csr};
    Q.data.csr.nnz = q_nnz;
    Q.data.csr.row_ptr = q_row;
    Q.data.csr.col_ind = q_col;
    Q.data.csr.vals = q_val;
    matrix_desc_t A = {.m = CONSTRAINTS, .n = VARIABLES, .fmt = matrix_csr};
    A.data.csr.nnz = a_nnz;
    A.data.csr.row_ptr = a_row;
    A.data.csr.col_ind = a_col;
    A.data.csr.vals = a_val;
    qp_problem_t *problem = create_qp_problem(c, &Q, NULL, NULL, &A, row_lower, row_upper,
                                             lower, upper, NULL, 0, NULL, NULL, NULL, 0, NULL);
    if (problem)
    {
        set_start_values(problem, x, y);
        for (int i = 0; i < CONSTRAINTS; ++i)
            problem->affine_cone_offset[i] = 0.02 * (i % 5 - 2);
    }
    return problem;
}

static pdhg_solver_state_t *make_state(qp_problem_t *problem, const pdhg_parameters_t *params)
{
    qp_problem_t *device_problem = convert_problem_to_device(problem, NULL);
    pdhg_solver_state_t *state = initialize_solver_state(params, problem, device_problem, NULL);
    free_device_problem(device_problem);
    check(state->inner_solver->mode == INNER_SOLVER_LINEARIZED, "inner solver selects linearized primal updates");
    check((state->inner_solver->linearized_context != NULL) == (strcmp(pdhcg_device_name(), "cuda") == 0),
          "inner solver initializes the device's linearized context");
    state->step_size = 0.13;
    state->primal_weight = 1.3;
    state->quadratic_objective_term->norm = 6.0;
    state->quadratic_objective_term->nonconvexity = 1.0;
    state->inner_count = 3;
    /* Ordinary device steps leave candidate storage to the evaluation step. */
    state->total_count = 1;
    state->is_this_major_iteration = false;
    sync_step_sizes_to_device(state);
    return state;
}

static void expected_first_step(const qp_problem_t *problem,
                                const pdhg_solver_state_t *state,
                                double rho,
                                double *next_x,
                                double *next_y,
                                double *reflected_x,
                                double *reflected_y)
{
    int n = problem->num_variables, m = problem->num_constraints;
    size_t primal_bytes = (size_t)(n > 0 ? n : 1) * sizeof(double);
    size_t dual_bytes = (size_t)(m > 0 ? m : 1) * sizeof(double);
    double *x = safe_malloc(primal_bytes), *x0 = safe_malloc(primal_bytes);
    double *y = safe_malloc(dual_bytes), *y0 = safe_malloc(dual_bytes);
    double *gradient = safe_malloc(primal_bytes);
    download(x, state->current_primal_solution, n);
    download(y, state->current_dual_solution, m);
    download(x0, state->initial_primal_solution, n);
    download(y0, state->initial_dual_solution, m);
    const CsrComponent *Q = problem->objective_sparse_matrix;
    const CsrComponent *A = problem->constraint_matrix;
    for (int i = 0; i < n; ++i)
    {
        gradient[i] = problem->objective_vector[i];
        for (int k = Q->row_ptr[i]; k < Q->row_ptr[i + 1]; ++k)
            gradient[i] += Q->val[k] * x[Q->col_ind[k]];
    }
    for (int i = 0; i < m; ++i)
        for (int k = A->row_ptr[i]; k < A->row_ptr[i + 1]; ++k)
            gradient[A->col_ind[k]] -= A->val[k] * y[i];
    /* The solver inflates the estimated spectral norm by its 0.998 safety factor. */
    double alpha = 1.0 / (state->quadratic_objective_term->norm / 0.998 +
                          state->primal_weight / state->step_size);
    double sigma = state->step_size * state->primal_weight;
    double weight = (double)(state->inner_count + 1) / (state->inner_count + 2);
    for (int i = 0; i < n; ++i)
    {
        double candidate = fmax(problem->variable_lower_bound[i],
                                fmin(x[i] - alpha * gradient[i], problem->variable_upper_bound[i]));
        reflected_x[i] = 2.0 * candidate - x[i];
        next_x[i] = weight * (rho * reflected_x[i] + (1.0 - rho) * x[i]) + (1.0 - weight) * x0[i];
    }
    for (int i = 0; i < m; ++i)
    {
        double value = problem->affine_cone_offset[i];
        for (int k = A->row_ptr[i]; k < A->row_ptr[i + 1]; ++k)
            value += A->val[k] * reflected_x[A->col_ind[k]];
        double projection = fmax(problem->constraint_lower_bound[i],
                                 fmin(value - y[i] / sigma, problem->constraint_upper_bound[i]));
        double candidate = y[i] - sigma * value + sigma * projection;
        reflected_y[i] = 2.0 * candidate - y[i];
        next_y[i] = weight * (rho * reflected_y[i] + (1.0 - rho) * y[i]) + (1.0 - weight) * y0[i];
    }
    free(x);
    free(x0);
    free(y);
    free(y0);
    free(gradient);
}

static void compare_states(const pdhg_solver_state_t *actual, const pdhg_solver_state_t *reference)
{
    int n = reference->num_variables, m = reference->num_constraints;
    int count = n > m ? n : m;
    if (count == 0)
        return;
    double *values = safe_malloc((size_t)count * sizeof(double));
    download(values, reference->current_primal_solution, n);
    check_vector("current primal", actual->current_primal_solution, values, n);
    download(values, reference->reflected_primal_solution, n);
    check_vector("reflected primal", actual->reflected_primal_solution, values, n);
    download(values, reference->current_dual_solution, m);
    check_vector("current dual", actual->current_dual_solution, values, m);
    download(values, reference->reflected_dual_solution, m);
    check_vector("reflected dual", actual->reflected_dual_solution, values, m);
    free(values);
}

static void check_execution(bool dense, int evaluation_frequency)
{
    qp_problem_t *problem = make_box_problem(dense);
    check(problem != NULL, "box problem construction");
    if (!problem)
        return;
    pdhg_parameters_t params;
    parameters_without_scaling(&params);
    params.termination_evaluation_frequency = evaluation_frequency;
    params.reflection_coefficient = 0.7;
    pdhg_solver_state_t *states[2];
    for (int i = 0; i < 2; ++i)
        states[i] = make_state(problem, &params);
    linearized_context_t *context = states[1]->inner_solver->linearized_context;
    if (strcmp(pdhcg_device_name(), "cuda") != 0)
    {
        check(context == NULL, "CPU keeps its ordinary implementation");
        printf("CUDA graph transitions (%s rows): skipped on %s\n",
                dense ? "dense" : "sparse", pdhcg_device_name());
        goto cleanup;
    }
    check(context != NULL, "CUDA execution context");
    if (!context)
        goto cleanup;

    double candidate_x[VARIABLES], candidate_y[CONSTRAINTS];
    for (int i = 0; i < VARIABLES; ++i)
        candidate_x[i] = 9.0 + 0.001 * i;
    for (int i = 0; i < CONSTRAINTS; ++i)
        candidate_y[i] = -8.0 - 0.001 * i;
    for (int i = 0; i < 2; ++i)
    {
        upload(states[i]->pdhg_primal_solution, candidate_x, VARIABLES);
        upload(states[i]->pdhg_dual_solution, candidate_y, CONSTRAINTS);
    }
    double expected_x[VARIABLES], expected_y[CONSTRAINTS], expected_rx[VARIABLES], expected_ry[CONSTRAINTS];
    expected_first_step(problem, states[0], params.reflection_coefficient,
                         expected_x, expected_y, expected_rx, expected_ry);
    const int batches[] = {1, 7, evaluation_frequency - 1, 3, evaluation_frequency - 1, 1};
    for (size_t batch = 0; batch < sizeof(batches) / sizeof(batches[0]); ++batch)
    {
        if (batch == 3)
        {
            for (int i = 0; i < 2; ++i)
            {
                states[i]->step_size = 0.09;
                states[i]->primal_weight = 0.6;
                states[i]->inner_count = 4;
                sync_step_sizes_to_device(states[i]);
                DEVICE_CHECK(pdhcg_device_copy(states[i]->initial_primal_solution, states[i]->current_primal_solution,
                                                VARIABLES * sizeof(double), PDHCG_COPY_DEVICE_TO_DEVICE));
                DEVICE_CHECK(pdhcg_device_copy(states[i]->initial_dual_solution, states[i]->current_dual_solution,
                                                CONSTRAINTS * sizeof(double), PDHCG_COPY_DEVICE_TO_DEVICE));
            }
        }
        for (int i = 0; i < batches[batch]; ++i)
        {
            states[0]->is_this_major_iteration = i == batches[batch] - 1;
            pdhg_update(states[0]);
            halpern_update(states[0], params.reflection_coefficient);
            ++states[0]->inner_count;
        }
        void *graph = context->graph;
        pdhcg_device_linearized_execute(states[1], &params, batches[batch]);
        if (batches[batch] == evaluation_frequency - 1)
            check(context->graph != NULL, "evaluation-period batch uses a CUDA graph");
        if (graph)
            check(context->graph == graph, "execution retains the same CUDA graph");
        else if (batches[batch] < evaluation_frequency - 1)
            check(context->graph == NULL, "partial batch does not create a CUDA graph");
        states[1]->inner_count += batches[batch];
        compare_states(states[1], states[0]);
        download(candidate_x, states[0]->pdhg_primal_solution, VARIABLES);
        download(candidate_y, states[0]->pdhg_dual_solution, CONSTRAINTS);
        check_vector("final primal candidate", states[1]->pdhg_primal_solution, candidate_x, VARIABLES);
        check_vector("final dual candidate", states[1]->pdhg_dual_solution, candidate_y, CONSTRAINTS);
        if (batch == 0)
        {
            for (int i = 0; i < 2; ++i)
            {
                check_vector("first-step primal oracle", states[i]->current_primal_solution, expected_x, VARIABLES);
                check_vector("first-step dual oracle", states[i]->current_dual_solution, expected_y, CONSTRAINTS);
                check_vector("first-step primal reflection", states[i]->reflected_primal_solution, expected_rx, VARIABLES);
                check_vector("first-step dual reflection", states[i]->reflected_dual_solution, expected_ry, CONSTRAINTS);
            }
        }
    }
cleanup:
    for (int i = 0; i < 2; ++i)
        pdhg_solver_state_free(states[i]);
    qp_problem_free(problem);
}

static void check_fixed_point_after_halpern(void)
{
    qp_problem_t *problem = make_box_problem(false);
    check(problem != NULL, "fixed-point problem construction");
    if (!problem)
        return;
    pdhg_parameters_t params;
    parameters_without_scaling(&params);
    params.reflection_coefficient = 0.7;
    pdhg_solver_state_t *state = make_state(problem, &params);
    double old_x[VARIABLES], old_y[CONSTRAINTS], dx[VARIABLES], dy[CONSTRAINTS];
    download(old_x, state->current_primal_solution, VARIABLES);
    download(old_y, state->current_dual_solution, CONSTRAINTS);
    state->fixed_point_error = -1.0;
    pdhg_iterations(state, &params, 1);
    check(state->fixed_point_error == -1.0, "iteration leaves fixed-point evaluation to the caller");

    /* Form the original reflected-minus-current direction using the saved
       pre-Halpern iterate, independently of the candidate reconstruction. */
    download(dx, state->reflected_primal_solution, VARIABLES);
    download(dy, state->reflected_dual_solution, CONSTRAINTS);
    double primal_sq = 0.0, dual_sq = 0.0, quadratic = 0.0, cross = 0.0;
    for (int i = 0; i < VARIABLES; ++i)
    {
        dx[i] -= old_x[i];
        primal_sq += dx[i] * dx[i];
    }
    for (int i = 0; i < CONSTRAINTS; ++i)
    {
        dy[i] -= old_y[i];
        dual_sq += dy[i] * dy[i];
    }
    const CsrComponent *Q = problem->objective_sparse_matrix;
    const CsrComponent *A = problem->constraint_matrix;
    for (int i = 0; i < VARIABLES; ++i)
        for (int k = Q->row_ptr[i]; k < Q->row_ptr[i + 1]; ++k)
            quadratic += dx[i] * Q->val[k] * dx[Q->col_ind[k]];
    for (int i = 0; i < CONSTRAINTS; ++i)
        for (int k = A->row_ptr[i]; k < A->row_ptr[i + 1]; ++k)
            cross += dy[i] * A->val[k] * dx[A->col_ind[k]];
    double expected = sqrt(fmax(0.0,
        state->primal_weight * primal_sq + dual_sq / state->primal_weight +
        state->step_size * ((state->quadratic_objective_term->norm / 0.998) * primal_sq - quadratic + 2.0 * cross)));

    compute_fixed_point_error(state);
    check_vector("post-Halpern primal direction", state->delta_primal_solution, dx, VARIABLES);
    check_vector("post-Halpern dual direction", state->delta_dual_solution, dy, CONSTRAINTS);
    check(expected > 0.0 && isfinite(state->fixed_point_error) &&
          fabs(state->fixed_point_error - expected) <= 2e-10 * (1.0 + expected),
          "post-Halpern fixed-point metric preserves its original scale and quadratic term");
    pdhg_solver_state_free(state);
    qp_problem_free(problem);
}

static void check_iteration_batches(bool dense, int evaluation_frequency)
{
    qp_problem_t *problem = make_box_problem(dense);
    check(problem != NULL, "batch problem construction");
    if (!problem)
        return;
    pdhg_parameters_t params;
    parameters_without_scaling(&params);
    params.termination_evaluation_frequency = evaluation_frequency;
    params.reflection_coefficient = 0.7;
    pdhg_solver_state_t *reference = make_state(problem, &params);
    pdhg_solver_state_t *actual = make_state(problem, &params);
    /* Keep the reference on ordinary steps while testing the shared entry point. */
    linearized_context_t *reference_context = reference->inner_solver->linearized_context;
    reference->inner_solver->linearized_context = NULL;
    check(strcmp(pdhcg_device_name(), "cuda") != 0 || actual->inner_solver->linearized_context != NULL,
          "CUDA batch uses its fused graph context");
    reference->best_primal_weight = actual->best_primal_weight = 1.3;

    const int batches[] = {1, 2, 7, evaluation_frequency - 1, evaluation_frequency, evaluation_frequency, 1};
    for (size_t batch = 0; batch < sizeof(batches) / sizeof(batches[0]); ++batch)
    {
        bool restart = batch == 2 || batch == 5 || batch == 6;
        if (restart)
        {
            compute_residual(reference, params.optimality_norm);
            compute_residual(actual, params.optimality_norm);
            perform_restart(reference, &params);
            perform_restart(actual, &params);
        }
        int old_total = actual->total_count;
        int old_inner = actual->inner_count;
        int iterations = batches[batch];
        double initial_error = actual->initial_fixed_point_error;
        /* Every batch executes its first step separately, even without restart. */
        pdhg_iterations(reference, &params, 1);
        pdhg_iterations(actual, &params, 1);
        --iterations;
        if (restart || iterations == 0)
        {
            compute_fixed_point_error(reference);
            compute_fixed_point_error(actual);
        }
        if (restart)
        {
            reference->initial_fixed_point_error = reference->fixed_point_error;
            initial_error = reference->fixed_point_error;
            actual->initial_fixed_point_error = actual->fixed_point_error;
        }
        if (iterations > 0)
        {
            pdhg_iterations(reference, &params, iterations);
            compute_fixed_point_error(reference);
            pdhg_iterations(actual, &params, iterations);
            compute_fixed_point_error(actual);
        }

        compare_states(actual, reference);
        double candidate[VARIABLES];
        download(candidate, reference->pdhg_primal_solution, VARIABLES);
        check_vector("batch primal candidate", actual->pdhg_primal_solution, candidate, VARIABLES);
        download(candidate, reference->pdhg_dual_solution, CONSTRAINTS);
        check_vector("batch dual candidate", actual->pdhg_dual_solution, candidate, CONSTRAINTS);
        check(actual->total_count == old_total + batches[batch] &&
              actual->total_count == reference->total_count, "batch total iteration count");
        check(actual->inner_count == old_inner + batches[batch] &&
              actual->inner_count == reference->inner_count, "batch inner iteration count");
        check(actual->is_this_major_iteration && reference->is_this_major_iteration,
              "batch ends with evaluation data");
        check(isfinite(actual->fixed_point_error) &&
              fabs(actual->fixed_point_error - reference->fixed_point_error) <=
                  2e-10 * (1.0 + fabs(reference->fixed_point_error)), "batch fixed-point error");
        check(isfinite(actual->initial_fixed_point_error) &&
              fabs(actual->initial_fixed_point_error - initial_error) <=
                  2e-10 * (1.0 + fabs(initial_error)), "restart first-step error is retained");
    }
    pdhg_solver_state_free(actual);
    reference->inner_solver->linearized_context = reference_context;
    pdhg_solver_state_free(reference);
    qp_problem_free(problem);
}

static void check_lowrank_execution(lowrank_middle_kind_t middle_type, bool with_sparse_q)
{
    enum { N = 3, M = 2, RANK = 2 };
    const double q[] = {0.8, 0.1, 0.0, 0.1, 0.6, 0.05, 0.0, 0.05, 0.4};
    const double r[] = {0.7, -0.2, 0.4, -0.1, 0.8, 0.3};
    double d[] = {1.0, 0.0, 0.0, 1.0};
    if (middle_type != PDHCG_D_NONE)
    {
        d[0] = 1.4;
        d[3] = 0.7;
    }
    if (middle_type == PDHCG_D_DENSE)
        d[1] = d[2] = 0.25;
    const double a[] = {1.0, -0.5, 0.2, -0.3, 0.4, 1.0};
    const double c[] = {-0.4, 0.7, -0.2};
    const double lower[] = {-0.6, -0.4, -0.5}, upper[] = {0.9, 0.8, 1.0};
    const double row_lower[] = {-0.2, -INFINITY}, row_upper[] = {0.4, 0.35};
    const double x[] = {0.5, -0.3, 0.6}, y[] = {0.15, -0.2};
    const double offset[] = {0.07, -0.04};
    double h[N * N];
    /* Form H independently, so the reference never uses low-rank products. */
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
        {
            h[i * N + j] = with_sparse_q ? q[i * N + j] : 0.0;
            for (int k = 0; k < RANK; ++k)
                for (int l = 0; l < RANK; ++l)
                    h[i * N + j] += r[k * N + i] * d[k * RANK + l] * r[l * N + j];
        }
    matrix_desc_t Q = {.m = N, .n = N, .fmt = matrix_dense};
    matrix_desc_t R = {.m = RANK, .n = N, .fmt = matrix_dense};
    matrix_desc_t D = {.m = RANK, .n = RANK, .fmt = matrix_dense};
    matrix_desc_t A = {.m = M, .n = N, .fmt = matrix_dense};
    matrix_desc_t H = {.m = N, .n = N, .fmt = matrix_dense};
    Q.data.dense.A = q;
    R.data.dense.A = r;
    D.data.dense.A = d;
    A.data.dense.A = a;
    H.data.dense.A = h;
    qp_problem_t *problem = create_qp_problem(c, with_sparse_q ? &Q : NULL, &R,
                                             middle_type == PDHCG_D_NONE ? NULL : &D, &A,
                                             row_lower, row_upper, lower, upper, NULL, 0, NULL, NULL, NULL, 0, NULL);
    qp_problem_t *explicit_problem = create_qp_problem(c, &H, NULL, NULL, &A, row_lower, row_upper,
                                                      lower, upper, NULL, 0, NULL, NULL, NULL, 0, NULL);
    check(problem && explicit_problem, "low-rank and explicit-H problem construction");
    if (!problem || !explicit_problem)
    {
        qp_problem_free(problem);
        qp_problem_free(explicit_problem);
        return;
    }
    set_start_values(problem, x, y);
    set_start_values(explicit_problem, x, y);
    for (int i = 0; i < M; ++i)
        problem->affine_cone_offset[i] = explicit_problem->affine_cone_offset[i] = offset[i];

    pdhg_parameters_t params;
    parameters_without_scaling(&params);
    params.termination_evaluation_frequency = 200;
    params.reflection_coefficient = 0.7;
    pdhg_solver_state_t *states[] = {make_state(explicit_problem, &params),
                                     make_state(problem, &params), make_state(problem, &params)};
    /* Compare both ordinary low-rank steps and the graph against explicit H. */
    linearized_context_t *reference_contexts[2];
    for (int i = 0; i < 2; ++i)
    {
        reference_contexts[i] = states[i]->inner_solver->linearized_context;
        states[i]->inner_solver->linearized_context = NULL;
    }
    for (int i = 0; i < 3; ++i)
        states[i]->best_primal_weight = 1.3;
    check(states[2]->quadratic_objective_term->lowrank_middle_type == (int)middle_type,
          "low-rank middle matrix uses the requested representation");
    check((states[2]->quadratic_objective_term->objective_sparse_matrix != NULL) == with_sparse_q,
          "low-rank execution covers the presence and absence of sparse Q");
    linearized_context_t *context = states[2]->inner_solver->linearized_context;
    double expected_x[N], expected_y[M], expected_rx[N], expected_ry[M];
    expected_first_step(explicit_problem, states[0], params.reflection_coefficient,
                         expected_x, expected_y, expected_rx, expected_ry);
    const int batches[] = {1, 200, 200, 7, 200};
    for (size_t batch = 0; batch < sizeof(batches) / sizeof(batches[0]); ++batch)
    {
        bool restart = batch == 2 || batch == 4;
        void *graph = context ? context->graph : NULL;
        for (int i = 0; i < 3; ++i)
        {
            if (restart)
            {
                compute_residual(states[i], params.optimality_norm);
                perform_restart(states[i], &params);
            }
            if (batch == 3)
            {
                states[i]->step_size = 0.09;
                states[i]->primal_weight = 0.6;
                sync_step_sizes_to_device(states[i]);
            }
            pdhg_iterations(states[i], &params, 1);
            if (restart)
            {
                compute_fixed_point_error(states[i]);
                states[i]->initial_fixed_point_error = states[i]->fixed_point_error;
            }
            if (batches[batch] > 1)
                pdhg_iterations(states[i], &params, batches[batch] - 1);
            compute_fixed_point_error(states[i]);
        }
        if (context && batches[batch] == params.termination_evaluation_frequency)
            check(context->graph != NULL, "low-rank evaluation batch creates its CUDA graph");
        if (graph)
            check(context->graph == graph, "low-rank graph is reused across restarts and step-size changes");
        for (int i = 1; i < 3; ++i)
        {
            compare_states(states[i], states[0]);
            double candidate[N];
            download(candidate, states[0]->pdhg_primal_solution, N);
            check_vector("low-rank primal candidate", states[i]->pdhg_primal_solution, candidate, N);
            download(candidate, states[0]->pdhg_dual_solution, M);
            check_vector("low-rank dual candidate", states[i]->pdhg_dual_solution, candidate, M);
            check(states[i]->inner_count == states[0]->inner_count &&
                  states[i]->total_count == states[0]->total_count, "low-rank batch iteration counts");
            check(isfinite(states[i]->fixed_point_error) &&
                  fabs(states[i]->fixed_point_error - states[0]->fixed_point_error) <=
                      2e-10 * (1.0 + fabs(states[0]->fixed_point_error)), "low-rank batch fixed-point error");
            if (restart)
                check(isfinite(states[i]->initial_fixed_point_error) &&
                      fabs(states[i]->initial_fixed_point_error - states[0]->initial_fixed_point_error) <=
                          2e-10 * (1.0 + fabs(states[0]->initial_fixed_point_error)),
                      "low-rank restart retains its first-step fixed-point error");
        }
        if (batch == 0)
            for (int i = 0; i < 3; ++i)
            {
                check_vector("low-rank first-step primal oracle", states[i]->current_primal_solution, expected_x, N);
                check_vector("low-rank first-step dual oracle", states[i]->current_dual_solution, expected_y, M);
                check_vector("low-rank first-step primal reflection", states[i]->reflected_primal_solution, expected_rx, N);
                check_vector("low-rank first-step dual reflection", states[i]->reflected_dual_solution, expected_ry, M);
            }
    }
    for (int i = 0; i < 3; ++i)
    {
        if (i < 2)
            states[i]->inner_solver->linearized_context = reference_contexts[i];
        pdhg_solver_state_free(states[i]);
    }
    qp_problem_free(problem);
    qp_problem_free(explicit_problem);
}

static void check_iteration_limits(void)
{
    const int rows[] = {0, 2, 4}, columns[] = {0, 1, 0, 1};
    const double values[] = {2.0, 1.0, 1.0, 2.0}, c[] = {-1.0, -0.5};
    matrix_desc_t Q = {.m = 2, .n = 2, .fmt = matrix_csr};
    Q.data.csr.nnz = 4;
    Q.data.csr.row_ptr = rows;
    Q.data.csr.col_ind = columns;
    Q.data.csr.vals = values;
    qp_problem_t *problem = create_qp_problem(c, &Q, NULL, NULL, NULL, NULL, NULL,
                                             NULL, NULL, NULL, 0, NULL, NULL, NULL, 0, NULL);
    check(problem != NULL, "iteration-limit problem construction");
    if (!problem)
        return;
    pdhg_parameters_t params;
    parameters_without_scaling(&params);
    params.termination_criteria.eps_optimal_relative = 1e-14;
    params.termination_criteria.eps_feasible_relative = 1e-14;
    params.termination_criteria.time_sec_limit = 20.0;
    const int cases[][2] = {{1, 0}, {200, 1}, {200, 2}, {1, 2}, {1, 3}, {2, 1}, {2, 2}, {2, 3},
                           {7, 8}, {200, 199}, {200, 200}, {200, 201}, {400, 399}, {400, 400}, {400, 401}};
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i)
    {
        params.termination_evaluation_frequency = cases[i][0];
        params.termination_criteria.iteration_limit = cases[i][1];
        pdhcg_result_t *result = solve_qp_problem(problem, &params);
        check(result != NULL, "iteration-limit solve returns a result");
        if (!result)
            continue;
        check(result->total_count <= cases[i][1], "evaluation batch respects iteration limit");
        if (result->termination_reason == TERMINATION_REASON_ITERATION_LIMIT)
            check(result->total_count == cases[i][1], "iteration limit is reached exactly");
        else
            check(result->termination_reason == TERMINATION_REASON_OPTIMAL,
                  "limited solve terminates optimally or at its iteration limit");
        if (cases[i][1] > 0)
            check(isfinite(result->primal_solution[0]) && fabs(result->primal_solution[0]) > 1e-10,
                  "partial batch returns a fresh candidate instead of the starting point");
        check(isfinite(result->cumulative_time_sec) && result->cumulative_time_sec >= 0.0,
              "final evaluation reports finite elapsed time");
        pdhcg_result_free(result);
    }
    qp_problem_free(problem);

    const int empty_row[] = {0, 0};
    const double linear[] = {-1.0}, lower[] = {0.0}, upper[] = {0.25};
    const double row_lower[] = {-INFINITY}, row_upper[] = {INFINITY};
    matrix_desc_t A = {.m = 1, .n = 1, .fmt = matrix_csr};
    A.data.csr.row_ptr = empty_row;
    problem = create_qp_problem(linear, NULL, NULL, NULL, &A, row_lower, row_upper,
                                 lower, upper, NULL, 0, NULL, NULL, NULL, 0, NULL);
    check(problem != NULL, "last-step optimality problem construction");
    if (!problem)
        return;
    /* Start at the primal optimum; the first LP prox computes its missing
       bound multiplier, so full optimality is first reached after one step. */
    set_start_values(problem, upper, NULL);
    params.termination_evaluation_frequency = 200;
    params.termination_criteria.iteration_limit = 1;
    pdhcg_result_t *result = solve_qp_problem(problem, &params);
    check(result && result->termination_reason == TERMINATION_REASON_OPTIMAL && result->total_count == 1 &&
          fabs(result->primal_solution[0] - 0.25) < 1e-12,
          "optimality on the last allowed step takes precedence over iteration limit");
    pdhcg_result_free(result);
    qp_problem_free(problem);
}

static void check_unconstrained_linearized(void)
{
    const int rows[] = {0, 2, 5, 7};
    const int columns[] = {0, 1, 0, 1, 2, 1, 2};
    const double values[] = {3.0, 0.5, 0.5, 2.0, 0.25, 0.25, 1.5};
    const double c[] = {-3.75, 0.9375, -0.25};
    const double lower[] = {-1.0, -0.5, -2.0}, upper[] = {1.0, 2.0, 2.0};
    /* At this point Qx+c=(-1,0.5,0): the first two coordinates are
       optimal at their upper/lower bounds and the third is interior. */
    const double expected[] = {1.0, -0.5, 0.25};
    matrix_desc_t Q = {.m = 3, .n = 3, .fmt = matrix_csr};
    Q.data.csr.nnz = 7;
    Q.data.csr.row_ptr = rows;
    Q.data.csr.col_ind = columns;
    Q.data.csr.vals = values;
    qp_problem_t *problem = create_qp_problem(c, &Q, NULL, NULL, NULL, NULL, NULL,
                                             lower, upper, NULL, 0, NULL, NULL, NULL, 0, NULL);
    check(problem != NULL, "unconstrained box QP construction");
    if (!problem)
        return;
    check(problem->num_constraints == 0, "box QP has no matrix constraint rows");
    pdhg_parameters_t params;
    parameters_without_scaling(&params);
    params.termination_evaluation_frequency = 100;
    params.termination_criteria.eps_optimal_relative = 1e-7;
    params.termination_criteria.eps_feasible_relative = 1e-7;
    params.termination_criteria.iteration_limit = 100000;
    params.termination_criteria.time_sec_limit = 20.0;
    pdhcg_result_t *result = solve_qp_problem(problem, &params);
    int passed = result && result->termination_reason == TERMINATION_REASON_OPTIMAL && result->total_inner_count == 0;
    if (passed)
        for (int i = 0; i < 3; ++i)
            passed &= isfinite(result->primal_solution[i]) && fabs(result->primal_solution[i] - expected[i]) < 1e-5;
    if (!passed)
        fprintf(stderr, "unconstrained linearized: status=%d, inner=%d\n",
                result ? (int)result->termination_reason : -1, result ? result->total_inner_count : -1);
    check(passed, "linearized box QP without constraint rows reaches its analytical optimum");
    pdhcg_result_free(result);
    qp_problem_free(problem);
}

static void check_inner_solver_modes(void)
{
    const double diagonal[] = {2.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 3.0};
    const double sparse[] = {2.0, 0.25, 0.0, 0.25, 4.0, 0.1, 0.0, 0.1, 3.0};
    const double c[] = {0.0, 0.0, 1.0};
    const int rows[] = {0, 0};
    const double lower[] = {-INFINITY}, upper[] = {INFINITY};
    matrix_desc_t A = {.m = 1, .n = 3, .fmt = matrix_csr};
    A.data.csr.row_ptr = rows;
    matrix_desc_t Q = {.m = 3, .n = 3, .fmt = matrix_dense};
    const non_diagonal_quadratic_mode_t modes[] = {
        NON_DIAGONAL_QUADRATIC_INNER, NON_DIAGONAL_QUADRATIC_LINEARIZED};
    const struct
    {
        const char *name;
        const double *quadratic;
        cone_spec_t cone;
        inner_solver_mode_t expected[2];
    } cases[] = {
        {"LP + PSD", NULL, {.type = CONE_PSD, .v_dim = 2}, {INNER_SOLVER_DIRECT, INNER_SOLVER_DIRECT}},
        {"diagonal Q + PSD", diagonal, {.type = CONE_PSD, .v_dim = 2},
         {INNER_SOLVER_LINEARIZED, INNER_SOLVER_LINEARIZED}},
        {"diagonal Q + SOC", diagonal, {.type = CONE_STANDARD_SOC, .v_dim = 1},
         {INNER_SOLVER_DIRECT, INNER_SOLVER_DIRECT}},
        {"non-diagonal Q + PSD", sparse, {.type = CONE_PSD, .v_dim = 2},
         {INNER_SOLVER_INNER_LOOP, INNER_SOLVER_LINEARIZED}}};

    /* Exercise the actual state initialization, including which workspaces
       are allocated, rather than checking the mode selector in isolation. */
    for (size_t case_index = 0; case_index < sizeof(cases) / sizeof(cases[0]); ++case_index)
    {
        Q.data.dense.A = cases[case_index].quadratic;
        qp_problem_t *problem = create_qp_problem(c, cases[case_index].quadratic ? &Q : NULL, NULL, NULL,
                                                 &A, lower, upper, NULL, NULL, NULL,
                                                 1, &cases[case_index].cone, NULL, NULL, 0, NULL);
        check(problem != NULL, "inner-solver dispatch problem construction");
        if (!problem)
            continue;
        for (size_t mode_index = 0; mode_index < sizeof(modes) / sizeof(modes[0]); ++mode_index)
        {
            pdhg_parameters_t params;
            parameters_without_scaling(&params);
            params.non_diagonal_quadratic_mode = modes[mode_index];
            qp_problem_t *device_problem = convert_problem_to_device(problem, NULL);
            pdhg_solver_state_t *state = initialize_solver_state(&params, problem, device_problem, NULL);
            free_device_problem(device_problem);
            inner_solver_mode_t expected = cases[case_index].expected[mode_index];
            if (state->inner_solver->mode != expected)
                fprintf(stderr, "inner-solver dispatch: %s, requested=%d mode=%d expected=%d\n",
                        cases[case_index].name, (int)modes[mode_index], (int)state->inner_solver->mode, (int)expected);
            check(state->inner_solver->mode == expected, "problem structure and mode select the inner solver");
            bool inner = expected == INNER_SOLVER_INNER_LOOP;
            check((state->inner_solver->bb_step_size != NULL) == inner &&
                  (state->inner_solver->primal_buffer != NULL) == inner &&
                  (state->cones.bb_primal_snapshot != NULL) == inner,
                  "only inner-loop updates allocate BB workspaces");
            pdhg_solver_state_free(state);
        }
        qp_problem_free(problem);
    }
}

static void check_diagonal_psd_step(void)
{
    const double sqrt_two = sqrt(2.0), sqrt_five = sqrt(5.0);
    const double q[] = {2.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 3.0};
    const double x[] = {1.0, 0.0, 2.0};
    const double unprojected[] = {-1.0, 2.0 * sqrt_two, 1.0};
    /* The unprojected matrix [-1, 2; 2, 1] has eigenvalues +/-sqrt(5).
       Its Euclidean PSD projection is (matrix + sqrt(5) I) / 2. */
    const double expected[] = {0.5 * (sqrt_five - 1.0), sqrt_two, 0.5 * (sqrt_five + 1.0)};
    double c[3], reflected[3], current[3];
    const double alpha = 1.0 / (4.0 / 0.998 + 1.6 / 0.2);
    for (int i = 0; i < 3; ++i)
    {
        c[i] = (x[i] - unprojected[i]) / alpha - q[3 * i + i] * x[i];
        reflected[i] = 2.0 * expected[i] - x[i];
        current[i] = 0.7 * expected[i] + 0.3 * x[i];
    }
    const int rows[] = {0, 0};
    const double lower[] = {-INFINITY}, upper[] = {INFINITY};
    matrix_desc_t A = {.m = 1, .n = 3, .fmt = matrix_csr};
    A.data.csr.row_ptr = rows;
    matrix_desc_t Q = {.m = 3, .n = 3, .fmt = matrix_dense};
    Q.data.dense.A = q;
    cone_spec_t cone = {.type = CONE_PSD, .start_idx = 0, .v_dim = 2};
    qp_problem_t *problem = create_qp_problem(c, &Q, NULL, NULL, &A, lower, upper,
                                             NULL, NULL, NULL, 1, &cone, NULL, NULL, 0, NULL);
    check(problem != NULL, "diagonal-Q PSD step problem construction");
    if (!problem)
        return;
    set_start_values(problem, x, NULL);
    pdhg_parameters_t params;
    parameters_without_scaling(&params);
    params.reflection_coefficient = 0.7;
    qp_problem_t *device_problem = convert_problem_to_device(problem, NULL);
    pdhg_solver_state_t *state = initialize_solver_state(&params, problem, device_problem, NULL);
    free_device_problem(device_problem);
    check(state->quadratic_objective_term->quad_obj_type == PDHCG_DIAG_Q,
          "PSD step retains a diagonal quadratic representation");
    check(state->inner_solver->mode == INNER_SOLVER_LINEARIZED,
          "diagonal-Q PSD variables select linearized projection");
    state->step_size = 0.2;
    state->primal_weight = 1.6;
    sync_step_sizes_to_device(state);
    pdhg_iterations(state, &params, 1);
    check_vector("diagonal-Q PSD candidate", state->pdhg_primal_solution, expected, 3);
    check_vector("diagonal-Q PSD reflection", state->reflected_primal_solution, reflected, 3);
    check_vector("diagonal-Q PSD Halpern iterate", state->current_primal_solution, current, 3);
    check(state->inner_solver->total_count == 0, "diagonal-Q PSD step performs no inner iterations");
    pdhg_solver_state_free(state);
    qp_problem_free(problem);
}

static void check_conic_linearized(cone_type_t type, bool affine)
{
    const double sqrt_two = sqrt(2.0);
    const int q_row[] = {0, 2, 5, 7};
    const int q_col[] = {0, 1, 0, 1, 2, 1, 2};
    const double q_val[] = {3.0, 0.5, 0.5, 2.0, 0.25, 0.25, 1.5};
    const int a_row[] = {0, 1};
    const int a_col[] = {0};
    const double a_val[] = {1.0};
    const double row_lower[] = {-INFINITY}, row_upper[] = {INFINITY};
    const int f_row[] = {0, 1, 2, 3};
    const int f_col[] = {0, 1, 2};
    const double f_val[] = {1.0, 1.0, 1.0};
    const double offset[] = {0.2, -0.1, 0.3};
    double target[] = {0.6, 0.8, 1.0};
    double gradient[] = {-0.3, -0.4, 0.5};
    if (type == CONE_PSD)
    {
        target[0] = target[2] = 1.0;
        target[1] = sqrt_two;
        gradient[0] = gradient[2] = 0.5;
        gradient[1] = -0.5 * sqrt_two;
    }
    double solution[3], c[3];
    for (int i = 0; i < 3; ++i)
        solution[i] = target[i] - (affine ? offset[i] : 0.0);
    for (int i = 0; i < 3; ++i)
    {
        c[i] = gradient[i];
        for (int k = q_row[i]; k < q_row[i + 1]; ++k)
            c[i] -= q_val[k] * solution[q_col[k]];
    }
    matrix_desc_t Q = {.m = 3, .n = 3, .fmt = matrix_csr};
    Q.data.csr.nnz = 7;
    Q.data.csr.row_ptr = q_row;
    Q.data.csr.col_ind = q_col;
    Q.data.csr.vals = q_val;
    matrix_desc_t A = {.m = 1, .n = 3, .fmt = matrix_csr};
    A.data.csr.nnz = 1;
    A.data.csr.row_ptr = a_row;
    A.data.csr.col_ind = a_col;
    A.data.csr.vals = a_val;
    matrix_desc_t F = {.m = 3, .n = 3, .fmt = matrix_csr};
    F.data.csr.nnz = 3;
    F.data.csr.row_ptr = f_row;
    F.data.csr.col_ind = f_col;
    F.data.csr.vals = f_val;
    cone_spec_t cone = {.type = type, .start_idx = 0, .v_dim = type == CONE_PSD ? 2 : 1};
    qp_problem_t *problem = create_qp_problem(c, &Q, NULL, NULL,
                                             affine ? NULL : &A, affine ? NULL : row_lower,
                                             affine ? NULL : row_upper, NULL, NULL, NULL,
                                             affine ? 0 : 1, affine ? NULL : &cone,
                                             affine ? &F : NULL, affine ? offset : NULL,
                                             affine ? 1 : 0, affine ? &cone : NULL);
    check(problem != NULL, "conic quadratic problem construction");
    if (!problem)
        return;
    pdhg_parameters_t params;
    parameters_without_scaling(&params);
    params.termination_evaluation_frequency = 10;
    params.termination_criteria.eps_optimal_relative = 1e-7;
    params.termination_criteria.eps_feasible_relative = 1e-7;
    params.termination_criteria.iteration_limit = 100000;
    params.termination_criteria.time_sec_limit = 20.0;
    pdhcg_result_t *result = solve_qp_problem(problem, &params);
    int passed = result && result->termination_reason == TERMINATION_REASON_OPTIMAL && result->total_inner_count == 0;
    if (passed)
        for (int i = 0; i < 3; ++i)
            passed &= fabs(result->primal_solution[i] - solution[i]) < 3e-4 * (1.0 + fabs(solution[i]));
    if (!passed)
    {
        fprintf(stderr, "%s %s linearized: status=%d, inner=%d",
                affine ? "affine" : "variable", type == CONE_PSD ? "PSD" : "SOC",
                result ? (int)result->termination_reason : -1, result ? result->total_inner_count : -1);
        if (result)
            fprintf(stderr, ", x=[%.9g %.9g %.9g]", result->primal_solution[0],
                    result->primal_solution[1], result->primal_solution[2]);
        fputc('\n', stderr);
    }
    check(passed, "linearized conic QP reaches its analytical optimum without inner iterations");
    pdhcg_result_free(result);
    qp_problem_free(problem);
}

int main(void)
{
    check_execution(false, 200);
    check_execution(true, 200);
    check_execution(false, 400);
    check_execution(true, 400);
    check_fixed_point_after_halpern();
    check_iteration_batches(false, 200);
    check_iteration_batches(true, 200);
    check_iteration_batches(false, 400);
    check_iteration_batches(true, 400);
    check_lowrank_execution(PDHCG_D_NONE, false);
    check_lowrank_execution(PDHCG_D_DIAG, true);
    check_lowrank_execution(PDHCG_D_DENSE, false);
    check_lowrank_execution(PDHCG_D_DENSE, true);
    check_iteration_limits();
    check_unconstrained_linearized();
    check_inner_solver_modes();
    check_diagonal_psd_step();
    check_conic_linearized(CONE_STANDARD_SOC, false);
    check_conic_linearized(CONE_STANDARD_SOC, true);
    check_conic_linearized(CONE_PSD, false);
    check_conic_linearized(CONE_PSD, true);
    printf("linearized execution: %s\n", failures ? "FAIL" : "PASS");
    return failures != 0;
}
