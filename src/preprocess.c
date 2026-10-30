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

#include "preprocess.h"
#include "distributed_interface.h"
#include "utils.h"

static void *copy_array_to_device(const void *source, size_t bytes)
{
    if (!source)
        return NULL;
    void *destination = NULL;
    DEVICE_CHECK(pdhcg_device_allocate(&destination, bytes));
    DEVICE_CHECK(pdhcg_device_copy(destination, source, bytes, PDHCG_COPY_HOST_TO_DEVICE));
    return destination;
}

static CsrComponent *copy_matrix_to_device(const CsrComponent *source, int rows, int nonzeros)
{
    if (!source)
        return NULL;
    CsrComponent *matrix = (CsrComponent *)safe_calloc(1, sizeof(CsrComponent));
    size_t row_bytes = ((size_t)rows + 1) * sizeof(int);
    matrix->row_ptr = (int *)copy_array_to_device(source->row_ptr, row_bytes);
    if (!matrix->row_ptr && nonzeros == 0)
    {
        DEVICE_CHECK(pdhcg_device_allocate((void **)&matrix->row_ptr, row_bytes));
        DEVICE_CHECK(pdhcg_device_zero(matrix->row_ptr, 0, row_bytes));
    }
    matrix->col_ind = (int *)copy_array_to_device(source->col_ind, (size_t)nonzeros * sizeof(int));
    matrix->val = (double *)copy_array_to_device(source->val, (size_t)nonzeros * sizeof(double));
    return matrix;
}

qp_problem_t *convert_problem_to_device(const qp_problem_t *problem, grid_context_t *grid_context)
{
    if (!problem)
        return NULL;
    qp_problem_t *device_problem = (qp_problem_t *)safe_malloc(sizeof(qp_problem_t));
    *device_problem = *problem;
    size_t variable_bytes = (size_t)problem->num_variables * sizeof(double);
    size_t constraint_bytes = (size_t)problem->num_constraints * sizeof(double);
    int q_rows = pdhcg_get_global_num_variables(grid_context);
    if (q_rows <= 0)
        q_rows = problem->num_variables;

    device_problem->variable_lower_bound =
        (double *)copy_array_to_device(problem->variable_lower_bound, variable_bytes);
    device_problem->variable_upper_bound =
        (double *)copy_array_to_device(problem->variable_upper_bound, variable_bytes);
    device_problem->objective_vector = (double *)copy_array_to_device(problem->objective_vector, variable_bytes);
    device_problem->constraint_lower_bound =
        (double *)copy_array_to_device(problem->constraint_lower_bound, constraint_bytes);
    device_problem->constraint_upper_bound =
        (double *)copy_array_to_device(problem->constraint_upper_bound, constraint_bytes);
    device_problem->affine_cone_offset = (double *)copy_array_to_device(problem->affine_cone_offset, constraint_bytes);
    device_problem->primal_start = (double *)copy_array_to_device(problem->primal_start, variable_bytes);
    device_problem->dual_start = (double *)copy_array_to_device(problem->dual_start, constraint_bytes);
    device_problem->constraint_matrix = copy_matrix_to_device(
        problem->constraint_matrix, problem->num_constraints, problem->constraint_matrix_num_nonzeros);
    device_problem->objective_sparse_matrix =
        copy_matrix_to_device(problem->objective_sparse_matrix, q_rows, problem->objective_sparse_matrix_num_nonzeros);
    device_problem->objective_lowrank_matrix = copy_matrix_to_device(problem->objective_lowrank_matrix,
                                                                     problem->num_rank_lowrank_obj,
                                                                     problem->objective_lowrank_matrix_num_nonzeros);
    device_problem->objective_lowrank_middle_matrix =
        copy_matrix_to_device(problem->objective_lowrank_middle_matrix,
                              problem->num_rank_lowrank_obj,
                              problem->objective_lowrank_middle_matrix_num_nonzeros);
    return device_problem;
}

static void free_device_matrix(CsrComponent *matrix)
{
    if (!matrix)
        return;
    DEVICE_CHECK(pdhcg_device_free(matrix->row_ptr));
    DEVICE_CHECK(pdhcg_device_free(matrix->col_ind));
    DEVICE_CHECK(pdhcg_device_free(matrix->val));
    free(matrix);
}

void free_device_problem(qp_problem_t *problem)
{
    if (!problem)
        return;
    free_device_matrix(problem->constraint_matrix);
    free_device_matrix(problem->objective_sparse_matrix);
    free_device_matrix(problem->objective_lowrank_matrix);
    free_device_matrix(problem->objective_lowrank_middle_matrix);
    DEVICE_CHECK(pdhcg_device_free(problem->variable_lower_bound));
    DEVICE_CHECK(pdhcg_device_free(problem->variable_upper_bound));
    DEVICE_CHECK(pdhcg_device_free(problem->objective_vector));
    DEVICE_CHECK(pdhcg_device_free(problem->constraint_lower_bound));
    DEVICE_CHECK(pdhcg_device_free(problem->constraint_upper_bound));
    DEVICE_CHECK(pdhcg_device_free(problem->affine_cone_offset));
    DEVICE_CHECK(pdhcg_device_free(problem->primal_start));
    DEVICE_CHECK(pdhcg_device_free(problem->dual_start));
    free(problem);
}

processed_qp_problem_t *preprocess_qp_problem(const qp_problem_t *source,
                                              const qp_problem_t *device_problem,
                                              quad_obj_type_t quad_type,
                                              int variable_start)
{
    if (source == NULL || device_problem == NULL)
        return NULL;

    processed_qp_problem_t *processed = (processed_qp_problem_t *)safe_calloc(1, sizeof(processed_qp_problem_t));

    processed->num_variables = device_problem->num_variables;
    processed->num_constraints = device_problem->num_constraints;
    processed->num_rank_lowrank_obj = device_problem->num_rank_lowrank_obj;
    processed->objective_constant = device_problem->objective_constant;
    processed->quad_type = quad_type;

    processed->constraint_matrix_num_nonzeros = device_problem->constraint_matrix_num_nonzeros;
    processed->objective_sparse_matrix_num_nonzeros = device_problem->objective_sparse_matrix_num_nonzeros;
    processed->objective_lowrank_matrix_num_nonzeros = device_problem->objective_lowrank_matrix_num_nonzeros;

    processed->variable_lower_bound = device_problem->variable_lower_bound;
    processed->variable_upper_bound = device_problem->variable_upper_bound;
    processed->objective_vector = device_problem->objective_vector;
    processed->constraint_lower_bound = device_problem->constraint_lower_bound;
    processed->constraint_upper_bound = device_problem->constraint_upper_bound;
    processed->primal_start = device_problem->primal_start;
    processed->dual_start = device_problem->dual_start;
    processed->constraint_matrix = device_problem->constraint_matrix;
    processed->objective_sparse_matrix = device_problem->objective_sparse_matrix;
    processed->objective_lowrank_matrix = device_problem->objective_lowrank_matrix;

    processed->objective_lowrank_middle_kind = PDHCG_D_NONE;
    processed->objective_lowrank_middle_diag = NULL;
    processed->objective_lowrank_middle_dense = NULL;
    {
        const CsrComponent *D_csr = source->objective_lowrank_middle_matrix;
        int D_nnz = source->objective_lowrank_middle_matrix_num_nonzeros;
        int k = source->num_rank_lowrank_obj;
        if (D_csr && D_csr->row_ptr && D_nnz > 0 && k > 0)
        {
            bool diag_only = true;
            for (int r = 0; r < k && diag_only; ++r)
            {
                for (int j = D_csr->row_ptr[r]; j < D_csr->row_ptr[r + 1]; ++j)
                {
                    if (D_csr->col_ind[j] != r)
                    {
                        diag_only = false;
                        break;
                    }
                }
            }

            if (diag_only)
            {
                processed->objective_lowrank_middle_kind = PDHCG_D_DIAG;
                ALLOC_ZERO(processed->objective_lowrank_middle_diag, (size_t)k * sizeof(double));
            }
            else
            {
                processed->objective_lowrank_middle_kind = PDHCG_D_DENSE;
                size_t rank2 = (size_t)k * (size_t)k;
                ALLOC_ZERO(processed->objective_lowrank_middle_dense, rank2 * sizeof(double));
            }
        }
    }

    if (processed->quad_type == PDHCG_DIAG_Q)
    {
        DEVICE_CHECK(pdhcg_device_allocate((void **)&processed->diagonal_quad_objective,
                                           (size_t)processed->num_variables * sizeof(double)));
    }

    pdhcg_device_preprocess_qp_problem(device_problem, variable_start, processed);
    return processed;
}

void free_processed_qp_problem(processed_qp_problem_t *processed)
{
    if (processed == NULL)
        return;

    if (processed->diagonal_quad_objective)
        DEVICE_CHECK(pdhcg_device_free(processed->diagonal_quad_objective));
    if (processed->objective_lowrank_middle_diag)
        DEVICE_CHECK(pdhcg_device_free(processed->objective_lowrank_middle_diag));
    if (processed->objective_lowrank_middle_dense)
        DEVICE_CHECK(pdhcg_device_free(processed->objective_lowrank_middle_dense));
    free(processed);
}
