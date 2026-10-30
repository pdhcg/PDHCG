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

void pdhcg_device_preprocess_qp_problem(const qp_problem_t *problem,
                                        int variable_start,
                                        processed_qp_problem_t *processed)
{
    if (processed->quad_type == PDHCG_DIAG_Q)
    {
        const CsrComponent *Q = problem->objective_sparse_matrix;
        for (int column = 0; column < problem->num_variables; ++column)
        {
            double value = 0.0;
            if (Q && Q->row_ptr && Q->col_ind && Q->val)
            {
                int row = variable_start + column;
                for (int index = Q->row_ptr[row]; index < Q->row_ptr[row + 1]; ++index)
                    value = Q->val[index];
            }
            processed->diagonal_quad_objective[column] = value;
        }
    }

    const CsrComponent *D = problem->objective_lowrank_middle_matrix;
    int rank = problem->num_rank_lowrank_obj;
    if (processed->objective_lowrank_middle_kind == PDHCG_D_DIAG)
    {
        for (int row = 0; row < rank; ++row)
            for (int index = D->row_ptr[row]; index < D->row_ptr[row + 1]; ++index)
                processed->objective_lowrank_middle_diag[row] = D->val[index];
    }
    else if (processed->objective_lowrank_middle_kind == PDHCG_D_DENSE)
    {
        for (int row = 0; row < rank; ++row)
            for (int index = D->row_ptr[row]; index < D->row_ptr[row + 1]; ++index)
                processed->objective_lowrank_middle_dense[(size_t)row * rank + D->col_ind[index]] = D->val[index];
    }
}
