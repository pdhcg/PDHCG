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

#include "pdhcg.h"
#include "preprocess.h"
#include "solver_state.h"

#include <math.h>
#include <stdio.h>

static int failures = 0;

static pdhg_solver_state_t *initialize_scaled_problem(const pdhg_parameters_t *parameters, const qp_problem_t *problem)
{
    pdhg_parameters_t params = *parameters;
    params.sv_max_iter = 2;
    params.verbose = 0;
    qp_problem_t *device_problem = convert_problem_to_device(problem, NULL);
    pdhg_solver_state_t *state = initialize_solver_state(&params, problem, device_problem, NULL);
    free_device_problem(device_problem);
    return state;
}

static void check_close(const char *name, double actual, double expected)
{
    if (!isfinite(actual) || fabs(actual - expected) > 1e-12 * fmax(1.0, fabs(expected)))
    {
        fprintf(stderr, "%s: got %.17g, expected %.17g\n", name, actual, expected);
        ++failures;
    }
}

static qp_problem_t *make_problem(int lowrank, int with_middle)
{
    static const int a_row[] = {0, 2};
    static const int a_col[] = {0, 1};
    static const double a_val[] = {1.0, 1.0};
    static const int q_row[] = {0, 2, 4};
    static const int q_col[] = {0, 1, 0, 1};
    static const double q_val[] = {16.0, 8.0, 8.0, 16.0};
    static const int r_row[] = {0, 2};
    static const int r_col[] = {0, 1};
    static const double r_val[] = {3.0, 4.0};
    static const int d_row[] = {0, 1};
    static const int d_col[] = {0};
    static const double d_val[] = {2.0};
    static const double zero[] = {0.0, 0.0};
    static const double con_lb[] = {-INFINITY};
    static const double con_ub[] = {INFINITY};

    matrix_desc_t A = {.m = 1, .n = 2, .fmt = matrix_csr};
    A.data.csr.nnz = 2;
    A.data.csr.row_ptr = a_row;
    A.data.csr.col_ind = a_col;
    A.data.csr.vals = a_val;

    matrix_desc_t Q = {.m = 2, .n = 2, .fmt = matrix_csr};
    Q.data.csr.nnz = 4;
    Q.data.csr.row_ptr = q_row;
    Q.data.csr.col_ind = q_col;
    Q.data.csr.vals = q_val;

    matrix_desc_t R = {.m = 1, .n = 2, .fmt = matrix_csr};
    R.data.csr.nnz = 2;
    R.data.csr.row_ptr = r_row;
    R.data.csr.col_ind = r_col;
    R.data.csr.vals = r_val;

    matrix_desc_t D = {.m = 1, .n = 1, .fmt = matrix_csr};
    D.data.csr.nnz = 1;
    D.data.csr.row_ptr = d_row;
    D.data.csr.col_ind = d_col;
    D.data.csr.vals = d_val;

    return create_qp_problem(zero,
                             lowrank ? NULL : &Q,
                             lowrank ? &R : NULL,
                             with_middle ? &D : NULL,
                             &A,
                             con_lb,
                             con_ub,
                             NULL,
                             NULL,
                             NULL,
                             0,
                             NULL,
                             NULL,
                             NULL,
                             0,
                             NULL);
}

static pdhg_parameters_t scaling_parameters(non_diagonal_quadratic_mode_t mode, int ruiz)
{
    pdhg_parameters_t params;
    set_default_parameters(&params);
    params.non_diagonal_quadratic_mode = mode;
    params.curtis_reid_iterations = 0;
    params.l_inf_ruiz_iterations = ruiz;
    params.has_pock_chambolle_alpha = !ruiz;
    params.pock_chambolle_alpha = 1.0;
    params.bound_objective_rescaling = false;
    return params;
}

static void check_scaling(int lowrank, int with_middle, int ruiz)
{
    qp_problem_t *problem = make_problem(lowrank, with_middle);
    pdhg_parameters_t params = scaling_parameters(NON_DIAGONAL_QUADRATIC_LINEARIZED, ruiz);
    pdhg_solver_state_t *state = initialize_scaled_problem(&params, problem);

    const double multiplier = with_middle ? 2.0 : 1.0;
    const double expected0 =
        lowrank ? (ruiz ? sqrt(12.0 * multiplier) : sqrt(1.0 + 21.0 * multiplier)) : (ruiz ? 4.0 : 5.0);
    const double expected1 =
        lowrank ? (ruiz ? sqrt(16.0 * multiplier) : sqrt(1.0 + 28.0 * multiplier)) : (ruiz ? 4.0 : 5.0);
    check_close(
        lowrank ? "low-rank variable scale 0" : "sparse variable scale 0", state->variable_rescaling[0], expected0);
    check_close(
        lowrank ? "low-rank variable scale 1" : "sparse variable scale 1", state->variable_rescaling[1], expected1);

    pdhg_solver_state_free(state);
    qp_problem_free(problem);
}

static void check_inner_unchanged(void)
{
    qp_problem_t *problem = make_problem(0, 0);
    pdhg_parameters_t params = scaling_parameters(NON_DIAGONAL_QUADRATIC_INNER, 1);
    pdhg_solver_state_t *state = initialize_scaled_problem(&params, problem);
    check_close("inner variable scale 0", state->variable_rescaling[0], 1.0);
    check_close("inner variable scale 1", state->variable_rescaling[1], 1.0);
    pdhg_solver_state_free(state);
    qp_problem_free(problem);
}

/* The final coefficients are representable even when applying the variable
   factors before the objective factor would overflow or underflow. */
static void check_deferred_range(int lowrank, int underflow)
{
    qp_problem_t *problem = make_problem(lowrank, 0);
    problem->constraint_matrix->val[0] = 1e-100;
    problem->constraint_matrix->val[1] = 1e100;
    problem->objective_vector[0] = underflow ? 0.0 : 1e100;
    problem->constraint_upper_bound[0] = underflow ? 1e100 : INFINITY;
    CsrComponent *matrix = lowrank ? problem->objective_lowrank_matrix : problem->objective_sparse_matrix;
    int rows = lowrank ? 1 : 2;
    for (int k = 0; k < matrix->row_ptr[rows]; ++k)
        matrix->val[k] = underflow ? 1e-300 : 1e300;

    pdhg_parameters_t params = scaling_parameters(NON_DIAGONAL_QUADRATIC_INNER, 0);
    params.curtis_reid_iterations = 1;
    params.has_pock_chambolle_alpha = false;
    params.bound_objective_rescaling = true;
    pdhg_solver_state_t *state = initialize_scaled_problem(&params, problem);
    const quadratic_objective_term_t *term = state->quadratic_objective_term;
    const device_sparse_matrix_csr_t *scaled = lowrank ? term->objective_lowrank_matrix : term->objective_sparse_matrix;
    long double multiplier = (long double)state->objective_vector_rescaling / state->constraint_bound_rescaling;
    if (lowrank)
        multiplier = sqrtl(multiplier);
    for (int row = 0; row < rows; ++row)
        for (int k = matrix->row_ptr[row]; k < matrix->row_ptr[row + 1]; ++k)
        {
            long double expected = (long double)matrix->val[k] * multiplier;
            expected /= state->variable_rescaling[matrix->col_ind[k]];
            if (!lowrank)
                expected /= state->variable_rescaling[row];
            double reference = (double)expected;
            if (!isfinite(reference) || reference == 0.0 || !isfinite(scaled->val[k]) ||
                fabs(scaled->val[k] / reference - 1.0) > 2e-12)
            {
                fprintf(stderr,
                        "deferred %s %s[%d]: got %.17g, expected %.17g\n",
                        lowrank ? "R" : "Q",
                        underflow ? "underflow" : "overflow",
                        k,
                        scaled->val[k],
                        reference);
                ++failures;
            }
        }
    pdhg_solver_state_free(state);
    qp_problem_free(problem);
}

int main(void)
{
    check_scaling(0, 0, 1);
    check_scaling(0, 0, 0);
    check_scaling(1, 0, 1);
    check_scaling(1, 0, 0);
    check_scaling(1, 1, 1);
    check_scaling(1, 1, 0);
    check_inner_unchanged();
    check_deferred_range(0, 0);
    check_deferred_range(0, 1);
    check_deferred_range(1, 0);
    check_deferred_range(1, 1);
    printf("%s\n", failures == 0 ? "PASS" : "FAIL");
    return failures == 0 ? 0 : 1;
}
