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

#include "checks.h"
#include "preprocess.h"
#include "utils.h"

/* Preserve the last stored entry when a CSR coordinate is repeated. */
__global__ static void finalize_middle_kernel(device_sparse_matrix_csr_t middle, double *output, bool diagonal)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= middle.num_rows)
        return;
    for (int index = middle.row_ptr[row]; index < middle.row_ptr[row + 1]; ++index)
    {
        size_t slot = diagonal ? (size_t)row : (size_t)row * middle.num_cols + middle.col_ind[index];
        output[slot] = middle.val[index];
    }
}

__global__ static void extract_diagonal_kernel(CsrComponent matrix, int columns, int start, double *diagonal)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= columns)
        return;
    /* DIAG_Q validation guarantees that all present entries belong to this diagonal. */
    double value = 0.0;
    for (int index = matrix.row_ptr[start + row]; index < matrix.row_ptr[start + row + 1]; ++index)
        value = matrix.val[index];
    diagonal[row] = value;
}

void pdhcg_device_preprocess_qp_problem(const qp_problem_t *problem,
                                        int variable_start,
                                        processed_qp_problem_t *processed)
{
    if (processed->quad_type == PDHCG_DIAG_Q && problem->num_variables > 0)
        extract_diagonal_kernel<<<((size_t)problem->num_variables + 255) / 256, 256>>>(
            *problem->objective_sparse_matrix,
            problem->num_variables,
            variable_start,
            processed->diagonal_quad_objective);

    if (processed->objective_lowrank_middle_kind != PDHCG_D_NONE)
    {
        int rank = problem->num_rank_lowrank_obj;
        const CsrComponent *source = problem->objective_lowrank_middle_matrix;
        device_sparse_matrix_csr_t middle = {rank,
                                             rank,
                                             problem->objective_lowrank_middle_matrix_num_nonzeros,
                                             source->row_ptr,
                                             source->col_ind,
                                             source->val};
        bool diagonal = processed->objective_lowrank_middle_kind == PDHCG_D_DIAG;
        double *output =
            diagonal ? processed->objective_lowrank_middle_diag : processed->objective_lowrank_middle_dense;
        finalize_middle_kernel<<<((size_t)rank + 255) / 256, 256>>>(middle, output, diagonal);
    }
    CUDA_CHECK(cudaGetLastError());
}
