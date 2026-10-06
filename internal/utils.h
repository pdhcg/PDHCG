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

#pragma once

#include "internal_types.h"
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#ifdef __cplusplus
extern "C"
{
#endif
    static const double HOST_ONE = 1.0;
    static const double HOST_ZERO = 0.0;

#define DEVICE_CHECK(call)                                                                                             \
    do                                                                                                                 \
    {                                                                                                                  \
        int status_ = (call);                                                                                          \
        if (status_ != 0)                                                                                              \
        {                                                                                                              \
            fprintf(stderr, "Device error at %s:%d (status %d)\n", __FILE__, __LINE__, status_);                       \
            exit(EXIT_FAILURE);                                                                                        \
        }                                                                                                              \
    } while (0)

#define ALLOC_AND_COPY(dest, src, bytes)                                                                               \
    DEVICE_CHECK(pdhcg_device_allocate((void **)&dest, bytes));                                                        \
    DEVICE_CHECK(pdhcg_device_copy(dest, src, bytes, PDHCG_COPY_HOST_TO_DEVICE));

#define ALLOC_AND_COPY_CSR(dest_csr, src_csr, n_rows, nnz)                                                             \
    do                                                                                                                 \
    {                                                                                                                  \
        ALLOC_AND_COPY((dest_csr)->row_ptr, (src_csr)->row_ptr, ((n_rows) + 1) * sizeof(int));                         \
                                                                                                                       \
        ALLOC_AND_COPY((dest_csr)->col_ind, (src_csr)->col_ind, (nnz) * sizeof(int));                                  \
                                                                                                                       \
        ALLOC_AND_COPY((dest_csr)->val, (src_csr)->val, (nnz) * sizeof(double));                                       \
    } while (0)

#define ALLOC_ZERO(dest, bytes)                                                                                        \
    DEVICE_CHECK(pdhcg_device_allocate((void **)&dest, bytes));                                                        \
    DEVICE_CHECK(pdhcg_device_zero(dest, 0, bytes));

    extern const double HOST_ONE;
    extern const double HOST_ZERO;

    void *safe_malloc(size_t size);

    void *safe_calloc(size_t num, size_t size);

    void *safe_realloc(void *ptr, size_t new_size);

    qp_problem_t *deepcopy_problem(const qp_problem_t *prob);

    qp_problem_t *create_problem_with_dummy_constraint(const qp_problem_t *prob);

    void compute_interaction_and_movement(pdhg_solver_state_t *solver_state, double *interaction, double *movement);

    bool should_do_adaptive_restart(pdhg_solver_state_t *solver_state,
                                    const restart_parameters_t *restart_params,
                                    int termination_evaluation_frequency);

    void check_termination_criteria(pdhg_solver_state_t *solver_state, const termination_criteria_t *criteria);

    void print_initial_info(const pdhg_parameters_t *params, const qp_problem_t *problem);

    void pdhg_final_log(const pdhcg_result_t *result, const pdhg_parameters_t *params);

    void display_iteration_stats(const pdhg_solver_state_t *solver_state, int verbose);

    const char *termination_reason_to_string(termination_reason_t reason);
    const char *problem_type_to_string(problem_type_t type);
    const char *quad_obj_type_to_string(quad_obj_type_t type);

    int get_print_frequency(int iter);

    void fill_or_copy(double **dest, int n, const double *src, double fill_value);

    int dense_to_csr(const matrix_desc_t *desc, int **row_ptr, int **col_ind, double **vals, int *nnz_out);

    int csc_to_csr(const matrix_desc_t *desc, int **row_ptr, int **col_ind, double **vals, int *nnz_out);

    int coo_to_csr(const matrix_desc_t *desc, int **row_ptr, int **col_ind, double **vals, int *nnz_out);

    void set_default_parameters(pdhg_parameters_t *params);

    double get_vector_sum(pdhcg_device_blas_t handle, int n, double *ones_d, const double *x_d);
    double get_vector_inf_norm(pdhcg_device_blas_t handle, int n, const double *x_d);

    CsrComponent *deepcopy_csr_component(const CsrComponent *src, size_t num_rows, size_t nnz);
    quad_obj_type_t detect_q_type(const CsrComponent *sparse_component,
                                  const CsrComponent *low_rank_component,
                                  int num_rows_sparse,
                                  int num_rows_low_rank);
    void ensure_objective_matrix_initialized(qp_problem_t *prob);
#ifdef __cplusplus
}

#endif
