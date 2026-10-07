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
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(condition)                                                                                                \
    do                                                                                                                 \
    {                                                                                                                  \
        if (!(condition))                                                                                              \
        {                                                                                                              \
            fprintf(stderr, "dispatch check failed at line %d: %s\n", __LINE__, #condition);                            \
            abort();                                                                                                   \
        }                                                                                                              \
    } while (0)

static int is_built(const char *device)
{
    size_t count = 0;
    const char *const *devices = pdhcg_get_built_devices(&count);
    CHECK(devices != NULL && count > 0);
    for (size_t i = 0; i < count; ++i)
        if (strcmp(device, devices[i]) == 0)
            return 1;
    return 0;
}

static void check_problem(const qp_problem_t *problem)
{
    CHECK(problem->num_variables == 2 && problem->num_constraints == 1);
    CHECK(problem->objective_vector[0] == -2.0 && problem->objective_vector[1] == -4.0);
    CHECK(problem->objective_sparse_matrix->val[0] == 2.0);
    CHECK(problem->objective_sparse_matrix->val[1] == 2.0);
    CHECK(problem->constraint_matrix->val[0] == 1.0 && problem->constraint_matrix->val[1] == 1.0);
    CHECK(problem->constraint_lower_bound[0] == 4.0);
    CHECK(problem->variable_lower_bound[0] == 0.0 && problem->variable_lower_bound[1] == 0.0);
    CHECK(problem->primal_start[0] == 0.0 && problem->primal_start[1] == 0.0);
    CHECK(problem->dual_start[0] == 0.0);
}

static void check_result(const pdhcg_result_t *result, double tolerance)
{
    CHECK(result != NULL);
    CHECK(result->termination_reason == TERMINATION_REASON_OPTIMAL);
    CHECK(result->num_variables == 2 && result->num_constraints == 1);
    CHECK(result->primal_solution != NULL && result->dual_solution != NULL);
    CHECK(fabs(result->primal_solution[0] - 1.5) < tolerance);
    CHECK(fabs(result->primal_solution[1] - 2.5) < tolerance);
    CHECK(fabs(result->primal_objective_value + 4.5) < tolerance);
}

static void check_rejected(qp_problem_t *problem, pdhg_parameters_t *params, const char *device)
{
    char error[256];
    params->device = device;
    CHECK(pdhcg_validate_parameters(params, error, sizeof(error)) != 0);
    CHECK(error[0] != '\0');
    CHECK(solve_qp_problem(problem, params) == NULL);
    CHECK(params->device == device);
    check_problem(problem);
}

int main(void)
{
    size_t count = 0;
    const char *const *devices = pdhcg_get_built_devices(&count);
    const char *default_device = pdhcg_get_default_device();
    CHECK(devices != NULL && count > 0 && default_device != NULL);
    CHECK(is_built(default_device));
    for (size_t i = 0; i < count; ++i)
    {
        CHECK(devices[i] != NULL && devices[i][0] != '\0');
        for (size_t j = 0; j < i; ++j)
            CHECK(strcmp(devices[i], devices[j]) != 0);
    }

    /* min x^2 + y^2 - 2x - 4y, x+y >= 4, x,y >= 0. All input
       arrays belong to the caller; the problem must own independent copies. */
    int q_rows[] = {0, 1, 2}, q_columns[] = {0, 1};
    int a_rows[] = {0, 2}, a_columns[] = {0, 1};
    double q_values[] = {2.0, 2.0}, a_values[] = {1.0, 1.0};
    double c[] = {-2.0, -4.0}, lower[] = {0.0, 0.0}, constraint_lower[] = {4.0};
    double start[] = {0.0, 0.0}, dual_start[] = {0.0};
    matrix_desc_t Q = {0}, A = {0};
    Q.m = Q.n = A.n = 2;
    A.m = 1;
    Q.fmt = A.fmt = matrix_csr;
    Q.data.csr.nnz = A.data.csr.nnz = 2;
    Q.data.csr.row_ptr = q_rows;
    Q.data.csr.col_ind = q_columns;
    Q.data.csr.vals = q_values;
    A.data.csr.row_ptr = a_rows;
    A.data.csr.col_ind = a_columns;
    A.data.csr.vals = a_values;
    qp_problem_t *problem = create_qp_problem(c, &Q, NULL, NULL, &A, constraint_lower, NULL,
                                             lower, NULL, NULL, 0, NULL, NULL, NULL, 0, NULL);
    CHECK(problem != NULL);
    set_start_values(problem, start, dual_start);
    check_problem(problem);

    pdhg_parameters_t params;
    char error[256];
    set_default_parameters(&params);
    CHECK(params.device == NULL || strcmp(params.device, "auto") == 0);
    CHECK(pdhcg_validate_parameters(&params, error, sizeof(error)) == 0);
    params.verbose = 0;
    params.presolve = false;
    params.num_threads = 2;
    params.termination_criteria.time_sec_limit = 20.0;
    params.termination_criteria.eps_optimal_relative = 1e-8;
    params.termination_criteria.eps_feasible_relative = 1e-8;

    const char *cuda_flag = getenv("PDHCG_TEST_CUDA");
    const int cuda_runtime = is_built("cuda") && cuda_flag && strcmp(cuda_flag, "1") == 0;
    const int default_runtime = strcmp(default_device, "cpu") == 0 || cuda_runtime;
    const char *selections[8];
    size_t num_selections = 0;
    if (is_built("cpu"))
        selections[num_selections++] = "cpu";
    if (cuda_runtime)
    {
        selections[num_selections++] = "cuda";
        if (is_built("cpu"))
            selections[num_selections++] = "cpu";
        selections[num_selections++] = "cuda";
    }
    if (default_runtime)
    {
        selections[num_selections++] = NULL;
        selections[num_selections++] = "auto";
    }
    if (is_built("cpu"))
        selections[num_selections++] = "cpu";

    pdhcg_result_t *results[8];
    for (size_t i = 0; i < num_selections; ++i)
    {
        params.device = selections[i];
        CHECK(pdhcg_validate_parameters(&params, error, sizeof(error)) == 0);
        results[i] = solve_qp_problem(problem, &params);
        check_result(results[i], 1e-5);
        check_problem(problem);
        CHECK(params.device == selections[i]);
        CHECK(strcmp(pdhcg_get_default_device(), default_device) == 0);

        /* Subsequent calls reuse the same problem after caller inputs change. */
        q_values[0] = q_values[1] = a_values[0] = a_values[1] = NAN;
        c[0] = c[1] = lower[0] = lower[1] = constraint_lower[0] = NAN;
        start[0] = start[1] = dual_start[0] = NAN;
        q_rows[1] = q_columns[0] = a_rows[1] = a_columns[0] = -1;
        check_problem(problem);
    }

    check_rejected(problem, &params, "unknown");
    check_rejected(problem, &params, "");
    check_rejected(problem, &params, "CPU");
    const char *known_devices[] = {"cpu", "cuda"};
    for (size_t i = 0; i < sizeof(known_devices) / sizeof(known_devices[0]); ++i)
        if (!is_built(known_devices[i]))
            check_rejected(problem, &params, known_devices[i]);

    pdhcg_result_t *default_result = NULL;
    if (default_runtime)
    {
        default_result = solve_qp_problem(problem, NULL);
        check_result(default_result, 1e-2);
        check_problem(problem);
    }
    qp_problem_free(problem);
    /* Results must not alias either the problem or another solve's storage. */
    for (size_t i = 0; i < num_selections; ++i)
    {
        check_result(results[i], 1e-5);
        pdhcg_result_free(results[i]);
    }
    if (default_result)
    {
        check_result(default_result, 1e-2);
        pdhcg_result_free(default_result);
    }
    printf("Public device dispatch passed (%zu backends, %zu explicit solves)\n", count, num_selections);
    return 0;
}
