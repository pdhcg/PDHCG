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

/*
 * create_qp_problem must reject a finite box bound on a cone slot.
 * Lift such variables manually with an auxiliary x_cone = x_box.
 */

#include "pdhcg.h"
#include "pdhcg_types.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(void)
{
    double aval[] = {1.0};
    int acol[] = {1};
    int arow[] = {0, 1};
    matrix_desc_t A = {0};
    A.m = 1;
    A.n = 3;
    A.fmt = matrix_csr;
    A.data.csr.nnz = 1;
    A.data.csr.row_ptr = arow;
    A.data.csr.col_ind = acol;
    A.data.csr.vals = aval;

    double c[] = {0.0, 0.0, 1.0};
    double con_lb[] = {4.0}, con_ub[] = {4.0};
    double var_lb[] = {0.0, -1e30, -1e30}; /* finite lower bound on v slot (index 0) */
    double var_ub[] = {1e30, 1e30, 1e30};

    cone_spec_t cones[] = {{.type = CONE_STANDARD_SOC, .start_idx = 0, .v_dim = 1, .is_fixed = NULL}};

    fprintf(stderr, "(expecting error on next line)\n");
    qp_problem_t *prob =
        create_qp_problem(c, NULL, NULL, NULL, &A, con_lb, con_ub, var_lb, var_ub, NULL, 1, cones, NULL, NULL, 0, NULL);
    int pass = (prob == NULL);
    if (prob)
        qp_problem_free(prob);
    printf("create_qp_problem returned %s -> %s\n", prob ? "non-NULL" : "NULL", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
