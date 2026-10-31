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

#include "device_general_op.h"
#include "device_kernels.h"
#include "pdhcg.h"
#include "pdhg_core_op.h"
#include "preconditioner.h"
#include "solver_state.h"
#include "utils.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures;

static double *device_copy(const double *values, int count)
{
    double *buffer = NULL;
    size_t bytes = (size_t)count * sizeof(double);
    DEVICE_CHECK(pdhcg_device_allocate((void **)&buffer, bytes));
    DEVICE_CHECK(pdhcg_device_copy(buffer, values, bytes, PDHCG_COPY_HOST_TO_DEVICE));
    return buffer;
}

static void compare_device(const char *label, const double *reference, const double *actual, int count)
{
    double *a = malloc((size_t)count * sizeof(double));
    double *b = malloc((size_t)count * sizeof(double));
    DEVICE_CHECK(pdhcg_device_copy(a, reference, (size_t)count * sizeof(double), PDHCG_COPY_DEVICE_TO_HOST));
    DEVICE_CHECK(pdhcg_device_copy(b, actual, (size_t)count * sizeof(double), PDHCG_COPY_DEVICE_TO_HOST));
    for (int i = 0; i < count; ++i)
    {
        if (!isfinite(b[i]) || fabs(a[i] - b[i]) > 2e-14 * fmax(1.0, fabs(a[i])))
        {
            fprintf(stderr, "%s[%d]: %.17g != %.17g\n", label, i, a[i], b[i]);
            ++failures;
            break;
        }
    }
    free(a);
    free(b);
}

/* Compare both kernels with free, one-sided and equality bounds. */
static void check_kernel(int n, int m, double weight, double reflection)
{
    int count = n > m ? n : m;
    double *values = malloc((size_t)count * sizeof(double));
    double *d[13];
    for (int slot = 0; slot < 13; ++slot)
    {
        for (int i = 0; i < count; ++i)
            values[i] = sin((double)(i + 1) * (slot + 1));
        if (slot == 8 || slot == 9)
        {
            for (int i = 0; i < count; ++i)
            {
                switch (i % 4)
                {
                    case 0: values[i] = slot == 8 ? -INFINITY : INFINITY; break;
                    case 1: values[i] = slot == 8 ? -0.3 : INFINITY; break;
                    case 2: values[i] = slot == 8 ? -INFINITY : 0.7; break;
                    case 3: values[i] = 0.2; break;
                }
            }
        }
        d[slot] = device_copy(values, count);
    }
    /* 0 initial x; 1 current x; 2 reflected x; 3 initial y; 4 current y;
       5 reflected y; 6 Ax; 7 offset; 8/9 bounds; 10/11/12 fused x/y/reflection. */
    DEVICE_CHECK(pdhcg_device_copy(d[10], d[1], (size_t)n * sizeof(double), PDHCG_COPY_DEVICE_TO_DEVICE));
    DEVICE_CHECK(pdhcg_device_copy(d[11], d[4], (size_t)m * sizeof(double), PDHCG_COPY_DEVICE_TO_DEVICE));
    const double step = 0.37;
    pdhcg_device_compute_next_pdhg_dual_solution(d[4], d[5], d[6], d[7], d[8], d[9], m, step);
    pdhcg_device_halpern_update(d[0], d[1], d[2], d[3], d[4], d[5], n, m, weight, reflection);
    pdhcg_device_dual_and_halpern_update(
        d[0], d[10], d[2], d[3], d[11], d[12], d[6], d[7], d[8], d[9], n, m, step, weight, reflection);
    compare_device("kernel primal", d[1], d[10], n);
    compare_device("kernel dual", d[4], d[11], m);
    compare_device("kernel reflected dual", d[5], d[12], m);
    for (int i = 0; i < 13; ++i)
        DEVICE_CHECK(pdhcg_device_free(d[i]));
    free(values);
}

/* Compare full iterations, including the major/print fallback paths. */
static void check_solver_updates(bool quadratic, bool affine)
{
    const int a_row[] = {0, 2, 4};
    const int a_col[] = {0, 1, 0, 1};
    const double a_val[] = {1.0, -0.3, 0.2, 1.0};
    const double obj[] = {0.3, -0.4};
    const double lb[] = {-2.0, -2.0}, ub[] = {2.0, 2.0};
    const double clb[] = {-1.0, 0.2}, cub[] = {1.0, 0.2};
    matrix_desc_t A = {.m = 2, .n = 2, .fmt = matrix_csr};
    A.data.csr.nnz = 4;
    A.data.csr.row_ptr = a_row;
    A.data.csr.col_ind = a_col;
    A.data.csr.vals = a_val;
    const int q_row[] = {0, 2, 4};
    const double q_val[] = {2.0, 0.5, 0.5, 1.0};
    matrix_desc_t Q = {.m = 2, .n = 2, .fmt = matrix_csr};
    Q.data.csr.nnz = 4;
    Q.data.csr.row_ptr = q_row;
    Q.data.csr.col_ind = a_col;
    Q.data.csr.vals = q_val;
    const int f_row[] = {0, 1, 2, 2}, f_col[] = {0, 1};
    const double f_val[] = {1.0, 1.0}, offset[] = {0.0, 0.0, 2.0};
    matrix_desc_t F = {.m = 3, .n = 2, .fmt = matrix_csr};
    F.data.csr.nnz = 2;
    F.data.csr.row_ptr = f_row;
    F.data.csr.col_ind = f_col;
    F.data.csr.vals = f_val;
    cone_spec_t cone = {.type = CONE_STANDARD_SOC, .start_idx = 0, .v_dim = 1};
    qp_problem_t *problem = create_qp_problem(obj, quadratic ? &Q : NULL, NULL, NULL, &A, clb, cub, lb, ub,
                                              NULL, 0, NULL, affine ? &F : NULL, affine ? offset : NULL,
                                              affine ? 1 : 0, affine ? &cone : NULL);
    if (!problem)
    {
        ++failures;
        return;
    }
    pdhg_parameters_t params;
    set_default_parameters(&params);
    params.verbose = 0;
    params.l_inf_ruiz_iterations = 0;
    params.curtis_reid_iterations = 0;
    params.has_pock_chambolle_alpha = false;
    params.bound_objective_rescaling = false;
    params.non_diagonal_quadratic_mode = NON_DIAGONAL_QUADRATIC_LINEARIZED;
    rescale_info_t *rescale = rescale_problem(&params, problem);
    pdhg_solver_state_t *reference = initialize_solver_state(&params, problem, rescale, NULL);
    pdhg_solver_state_t *fused = initialize_solver_state(&params, problem, rescale, NULL);
    initialize_step_size_and_primal_weight(reference, &params);
    /* Spectral estimates use randomized vectors. Use identical steps in both paths. */
    fused->step_size = reference->step_size;
    fused->primal_weight = reference->primal_weight;
    fused->quadratic_objective_term->norm = reference->quadratic_objective_term->norm;
    fused->quadratic_objective_term->nonconvexity = reference->quadratic_objective_term->nonconvexity;
    for (int iteration = 0; iteration < 300; ++iteration)
    {
        reference->total_count = fused->total_count = iteration;
        reference->inner_count = fused->inner_count = iteration;
        reference->is_this_major_iteration = fused->is_this_major_iteration = iteration % 40 == 39;
        pdhg_update(reference);
        halpern_update(reference, 0.8);
        pdhg_update_and_halpern(fused, 0.8);
        compare_device("solver primal", reference->current_primal_solution, fused->current_primal_solution, 2);
        compare_device("solver dual", reference->current_dual_solution, fused->current_dual_solution, reference->num_constraints);
        compare_device("solver reflected dual", reference->reflected_dual_solution, fused->reflected_dual_solution, reference->num_constraints);
        if (reference->is_this_major_iteration || ((iteration + 2) % get_print_frequency(iteration + 2)) == 0)
            compare_device("solver saved candidate", reference->pdhg_dual_solution, fused->pdhg_dual_solution, reference->num_constraints);
    }
    pdhg_solver_state_free(reference);
    pdhg_solver_state_free(fused);
    rescale_info_free(rescale);
    qp_problem_free(problem);
}

int main(void)
{
    DEVICE_CHECK(pdhcg_device_initialize());
    check_kernel(17, 33, 0.5, 1.0);
    check_kernel(33, 17, 0.99, 0.6);
    check_kernel(1025, 1100, 0.99, 0.8);
    check_solver_updates(false, false);
    check_solver_updates(true, false);
    check_solver_updates(false, true);
    if (failures)
        return 1;
    printf("Fused dual/Halpern update matches separate updates on %s.\n", pdhcg_device_name());
    return 0;
}
